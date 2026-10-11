"""
ss_pilot.py — a trained special-stage agent playing a giant ring inside a
main-game episode (coordinator.py with MG_SS=pilot).

The agent plays through the same SpecialStageEnv it trained on, attached to
the main game's emulator (stable_retro allows one per process), with the
settings of its train_ss_multistage.py run read from that file's RUNS table.
When the stage ends, the game shows its results and puts Sonic back at the
ring by itself; coordinator.py waits for that and carries on with the act.

THE HANDOVER (measured 2026-10-11 on a real ring entry: AIZ1, x 7,134, the
game's first special stage = board 1; no input): game mode $34 and the entry
flag ($FE48 = 1) come on together on the entry frame. The board is loaded 133
frames later and then matches the hand-captured curriculum_stage_1.state (102
spheres, start cell, standing, facing north). Up is first taken at entry+157;
at +156 it is ignored and Sonic sets off by himself at +349. But the speed-up
timer (an agent input) runs from the board's loading, so the frame Sonic sets
off on moves every speed-up along his run, and the 14 boards were captured by
hand at different moments. So the pilot waits until the timer reads what the
board's captured state reads, and hands over one frame before that: the env's
reset frame and its Up tap then play exactly as in training. Checked on that
entry: Tyler's own stage-1 inputs clear 102/102 in his 2,925 frames, and d9's
agent (deterministic) sees identical observations and makes identical choices
for all 1,670 decisions of its run, against the captured state. (Started at
the earliest frame instead, the same inputs died at sphere 66.)

ON PURPOSE, unlike training:
  - no stall rule: 600 frames without a pickup is a training device; in the
    game Sonic is still running. env._replaying switches it off, and with it
    the training episode log and the restart-pool snapshots;
  - SS_MAX_STEPS (18,000 frames) still ends the agent's turn; the pilot then
    lets go (no input) until the game ends the stage itself (Sonic runs into
    his own trail), within TAIL_CAP frames.
The board is named by its layout against the 14 captured states (None if it
matches none of them).
"""
import ast
import os
import sys

import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
GM_STAGE = 0x34             # Game_mode: special stage
HANDOVER_CAP = 900          # entry -> the stage takes Up (measured: 157 frames)
TAIL_CAP = 6000             # the agent's turn ended without the game ending it
LET_GO_TAP = 90             # ...then a tap of left this often (straight runs can loop forever)
SHRINE_ZONE = 0x17          # Hidden Palace as a giant ring's destination
SHRINE_CAP = 1500           # ring -> a gray Super Emerald touched (measured: 83 + 132-241)
GRAY_EMERALD = 0x0908de     # object code: a gray Super Emerald (subtype +$2C = its slot);
                            # a collected one becomes $0908de -> $09089e
# Emerald slot ($FFB2 + slot) -> the board it starts, by our curriculum_stage_N
# numbering (measured 2026-10-11 by touching each one; boards named by layout)
SUPER_SLOT_BOARD = {0: 11, 1: 10, 2: 12, 3: 9, 4: 13, 5: 8, 6: 14}
SS_KEYS = ('SS_ENV_VERSION', 'SS_USE_PIXELS', 'SS_PATCH_R', 'SS_DROP_POSITION',
           'SS_POINTER', 'SS_SPEED_BONUS')


def run_preset(run):
    """RUNS[run]['obs'] from train_ss_multistage.py, read with ast (importing
    the trainer would parse argv and start a run)."""
    tree = ast.parse(open(os.path.join(HOME, 'train_ss_multistage.py')).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'RUNS' for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == run:
                    obs = [x.value for x in v.keywords if x.arg == 'obs'][0]
                    return {x.arg: ast.literal_eval(x.value) for x in obs.keywords}
    raise KeyError(f'no run {run!r} in train_ss_multistage.RUNS')


def import_env(preset):
    """special_stage_env configured as the run was (it reads SS_* at import)."""
    if 'special_stage_env' in sys.modules:
        sse = sys.modules['special_stage_env']
        want = (int(preset.get('SS_ENV_VERSION', '2')), preset.get('SS_USE_PIXELS', '0') == '1',
                int(preset.get('SS_PATCH_R', '7')), preset.get('SS_POINTER', '0') == '1')
        have = (sse.ENV_VERSION, sse.USE_PIXELS, sse.PATCH_R, sse.POINTER)
        if want != have:
            raise RuntimeError(f'special_stage_env is already imported with {have}, the pilot needs {want}')
        return sse
    for k in SS_KEYS:
        os.environ.pop(k, None)
    os.environ.update(preset)
    os.environ.update(SS_STAGES='', SS_DEMOS='', SS_RESTART_PROB='0', SS_LOG=os.devnull)
    import special_stage_env as sse
    return sse


class SSPilot:
    def __init__(self, base_env, run='d9', model=None, deterministic=False, device='cpu',
                 match_timer=True):
        self.run = run
        self.sse = sse = import_env(run_preset(run))
        self.emu = base_env
        self.env = sse.SpecialStageEnv(base_env=base_env)
        self.env._replaying = True      # real play: no stall rule, no training log, no snapshots
        self.model_path = model or os.path.join(HOME, f'sonic_ss_multistage_{run}.zip')
        from sb3_contrib import RecurrentPPO
        self.model = RecurrentPPO.load(self.model_path, device=device)
        mo, eo = self.model.observation_space, self.env.observation_space
        if ({k: s.shape for k, s in mo.spaces.items()} != {k: s.shape for k, s in eo.spaces.items()}
                or self.model.action_space.n != self.env.action_space.n):
            raise RuntimeError(f'{self.model_path} does not fit run {run}\'s env: '
                               f'{mo} / {self.model.action_space} vs {eo} / {self.env.action_space}')
        self.deterministic = deterministic
        self.match_timer = match_timer
        self.NOOP = sse.ACTION_BUTTONS[0]
        self.UP = sse.ACTION_BUTTONS[4]
        self._boards = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _frame(self, buttons):
        _, _, _, _, info = self.emu.step(buttons)
        return info

    def _board_layouts(self):
        """Each captured board's (grid, speed timer) after the env's reset
        frame (computed once; the emulator's state is put back afterwards)."""
        if self._boards is None:
            here = self.emu.em.get_state()
            self._boards = {}
            for s in range(1, 15):
                path = self.sse.stage_file(s)
                if os.path.exists(path):
                    self.env.reset(options={'stage_file': path, 'replay': True})
                    self._boards[s] = (self.env._read_grid().copy(),
                                       int(self.emu.data.lookup_value('ss_speed_timer')))
            self.emu.em.set_state(here)
        return self._boards

    def _which_board(self, grid):
        return next((s for s, (g, _) in self._board_layouts().items() if np.array_equal(g, grid)), None)

    def _first_up(self):
        """From the entry frame, no input until the stage first takes Up. Each
        frame is tried twice from a saved state, with Up and with nothing: Up
        is taken when it changes Sonic's speed. (Before the board is loaded the
        stage's RAM still holds the level's leftovers, e.g. the sphere count
        read 5,120 or 0 depending on the run, so no single value can say the
        board is ready; the input test can.) Returns (state one frame before
        that frame, frames stepped from the entry frame to that state)."""
        prev = None
        for n in range(HANDOVER_CAP):
            here = self.emu.em.get_state()
            tried = self._frame(self.UP)
            up = int(tried['ss_move_speed'])
            self.emu.em.set_state(here)                 # undo the try
            info = self._frame(self.NOOP)
            if (up != int(info['ss_move_speed']) and up != 0 and int(info['special_stage'])
                    and all(int(i.get('mg_game_mode', GM_STAGE)) & 0x7F == GM_STAGE
                            for i in (info, tried))):
                if prev is None:
                    raise RuntimeError('the stage took Up on the frame after the ring')
                return prev, n - 1
            prev = here
        raise RuntimeError(f'the stage never took Up within {HANDOVER_CAP} frames of the ring')

    def handover(self):
        """-> (state to hand the env, frames from the entry frame to it, board).
        The speed-up timer (ss_speed_timer, 1,800 frames to the next speed
        level, an agent input) runs from the board's loading, not from the
        run's start, so the frame Sonic sets off on decides where every speed-up
        falls. The 14 boards were captured by hand at different moments
        (1,694 on board 1 vs 1,787 at its earliest start), and the agent learned
        each board with its own timing: Tyler's stage-1 inputs died at sphere 66
        when started 93 frames early. So with match_timer the pilot waits
        until the timer reads what the board's captured state reads after the
        env's reset frame (11-110 frames, inside the stage, the act clock
        stopped), and the agent gets exactly the stage it trained on."""
        prev, pre = self._first_up()
        env = self.env
        env.reset(options={'state': prev, 'replay': True})      # its no-op frame
        grid = env._read_grid()
        if env._ram_swapped is None or int(env._stage_total) != int((grid == 2).sum()):
            raise RuntimeError(f'handover: the sphere count ({env._stage_total}) does not match '
                               f'the board ({int((grid == 2).sum())} blue cells)')
        board = self._which_board(grid)
        if board is None or not self.match_timer:
            return prev, pre, board
        want = self._board_layouts()[board][1]
        state = prev
        for k in range(HANDOVER_CAP):
            self.emu.em.set_state(state)
            timer = int(self._frame(self.NOOP)['ss_speed_timer'])
            if timer == want:
                return state, pre + k, board
            if timer < want:
                raise RuntimeError(f'board {board}: the speed timer passed {want} ({timer}) '
                                   f'before the stage took Up')
            state = self.emu.em.get_state()
        raise RuntimeError(f'board {board}: the speed timer never reached {want}')

    # ── the stage ────────────────────────────────────────────────────────

    def play(self, script=None, trace=None):
        """Call on the entry frame (the main game's first frame in the stage).
        Plays the stage until the game ends it (clear: board emptied; death:
        the exit sequence starts). script: per-frame action indices to play
        instead of the agent (tests: a recording of Tyler's). trace: a list to
        append (observation, action) to at every decision. Returns a dict."""
        state, pre, board = self.handover()
        env = self.env
        if script is not None:
            env.reset(options={'state': state, 'stage': board or 0, 'replay': True})
            done = trunc = False
            for a in script:
                _, _, done, trunc, _ = env._step_frames(int(a), 1)
                if done or trunc:
                    break
            decisions = None
        else:
            obs, _ = env.reset(options={'state': state, 'stage': board or 0})
            lstm, start, decisions = None, True, 0
            while True:
                action, lstm = self.model.predict(obs, state=lstm, episode_start=np.array([start]),
                                                  deterministic=self.deterministic)
                start = False
                a = int(np.asarray(action).reshape(-1)[0])
                if trace is not None:
                    trace.append(({k: v.copy() for k, v in obs.items()}, a))
                obs, _, done, trunc, _ = env.step(a)
                decisions += 1
                if done or trunc:
                    break
        frames = pre + 1 + env.steps            # +1: the reset frame
        outcome = 'clear' if env._board_cleared else 'death' if done else 'timeout'
        tail = 0
        if outcome == 'timeout' or (script is not None and not done):
            while int(env._ram()[self.sse.ADDR_EXIT_TIMER]) == 0 and tail < TAIL_CAP:
                info = self._frame(self.sse.ACTION_BUTTONS[1] if tail % LET_GO_TAP == LET_GO_TAP - 1
                                   else self.NOOP)
                tail += 1
                if not int(info['special_stage']):
                    break
        return dict(board=board, outcome=outcome, spheres=int(env._spheres_collected),
                    total=int(env._stage_total), frames=frames + tail, to_run=pre + 1,
                    decisions=decisions, let_go=tail)

    # ── Hidden Palace: the Super Emeralds ─────────────────────────────────

    def _word_ram(self):
        r = np.frombuffer(bytes(self.emu.get_ram()), dtype=np.uint8)
        sw = r.reshape(-1, 2)[:, ::-1].reshape(-1)     # 16-bit words, host order
        return sw

    def gray_emeralds(self):
        """{slot: (x, y)} of the gray Super Emeralds in the object table."""
        r = self._word_ram()

        def word(i):
            return (int(r[i]) << 8) | int(r[i + 1])
        return {int(r[s + 0x2C]): (word(s + 0x10), word(s + 0x14))
                for s in range(0xB000, 0xCFCC, 0x4A)
                if ((word(s) << 16) | word(s + 2)) == GRAY_EMERALD}

    def shrine(self, prefer, buttons):
        """Call on the frame a giant ring has sent Sonic to Hidden Palace (all 7
        Chaos Emeralds; the entry flag reads 2 and the zone $17). Measured
        2026-10-11 (MHZ1's ring at (8768,1088), the emeralds granted the way
        the game stores them: count 7, $FFB2-$FFB8 = 2 for gray): the shrine's
        level runs 83 frames later, Sonic standing under the Master Emerald, act
        clock stopped, a gray Super Emerald on each pedestal; touching one
        starts ITS board (SUPER_SLOT_BOARD; the choice is free); after the
        stage the game returns to the act at the ring, not to the shrine.
        Walks under the most preferred gray emerald (prefer: boards in order)
        and jumps, steering to it, until the stage starts. buttons: the main
        game's {name: mask}. Returns (frames stepped, slot, expected board);
        the next call is play(), on the stage's entry frame."""
        target = slot = info = None
        held = False
        for f in range(SHRINE_CAP):
            if target is None and info is not None and int(info['mg_game_mode']) == 0x0C:
                gray = self.gray_emeralds()
                order = [sl for b in prefer for sl, bb in SUPER_SLOT_BOARD.items() if bb == b and sl in gray]
                if order:
                    slot = order[0]
                    target = gray[slot]
                elif gray:
                    slot = min(gray)
                    target = gray[slot]
            if target is None:
                btn = 'noop'
            else:
                dx = target[0] - int(info['x'])
                side = 'right' if dx > 0 else 'left'
                if not int(info['mg_status']) & 2:          # on the ground
                    if abs(dx) > 3:
                        btn, held = side, False
                    else:
                        btn = 'noop' if held else 'jump'    # a fresh press each time
                        held = not held
                else:
                    btn = f'jump+{side}' if abs(dx) > 2 else 'jump'
            _, _, _, _, info = self.emu.step(buttons[btn])
            if int(info['mg_game_mode']) & 0x7F == GM_STAGE:
                return f + 1, slot, SUPER_SLOT_BOARD.get(slot)
        raise RuntimeError(f'no Super Emerald touched within {SHRINE_CAP} frames of the shrine '
                           f'(gray: {self.gray_emeralds()})')
