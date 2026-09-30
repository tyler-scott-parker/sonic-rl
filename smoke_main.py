#!/usr/bin/env python3
"""
smoke_main.py — pre-launch checks for the base game (coordinator.py), and the
maker of the synthetic mini-act the `smoke` run trains on.

  python3 smoke_main.py

Each check runs against the real emulator:
  1. the RAM map reads what it should at the Angel Island 1 start
  2. a death is seen on the frame it happens (routine 6), while the lives
     counter still reads 3
  3. holding down with left/right doesn't roll, and down alone does — the
     premise of record_play.keys_to_action — and the mapping table itself
  4. a scripted mini-act (AIZ1 to a test signpost at x=7100) clears on the
     frame the in-game timer stops, and replays frame for frame; written to
     demos/_synthetic_aiz1_goal7100.npz for `train_main.py smoke`
  5. from EVERY start point, the demo's own inputs clear on the demo's exact
     frame and return exactly CLEAR_BONUS ("0 frames faster than the demo"):
     the start states, observation stacks and reward accounting agree
  6. the recorder, driven headless: record with a rewind, save, resume from
     its cache, extend, save — and the file replays
  7. every playthrough recorded in demos/ still replays (fixture off)
"""
import os
import sys
import glob

os.environ['MG_TEST_GOAL_X'] = '7100'          # before coordinator is imported

import numpy as np                              # noqa: E402
import coordinator as C                         # noqa: E402
import demos                                    # noqa: E402
import record_play                              # noqa: E402

SYNTH = os.path.join(demos.DEMO_DIR, '_synthetic_aiz1_goal7100.npz')
# Button changes only on frames = 1 (mod 4): start points sit on frames = 0
# (mod 4), so from any of them the script is expressible as 4-frame steps.
SCRIPT = [2] * 401 + [7] * 40 + [2] * 200       # run right, jump the ledge
failures = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}", flush=True)
    if not ok:
        failures.append(what)


def script_action(i):
    return SCRIPT[i] if i < len(SCRIPT) else 0


def ram_map():
    print('1. RAM map at the Angel Island 1 start')
    env = C.make_retro()
    env.reset()
    env.em.set_state(C.integration_state(C.DEFAULT_START))
    _, info = C.emu_frame(env, C.ACTION_BUTTONS[0])
    want = dict(zone=0, act=0, mg_game_mode=0x0C, mg_routine=2, mg_entry=0,
                mg_boss=0, mg_apparent_act=0, lives=3)
    got = {k: int(info[k]) for k in want}
    check(got == want, f'{got}')
    check(int(info['mg_hud_timer']) != 0 and C.frame_flags(info) == (True, False, False, True),
          f"timer running at {C.fmt_time(C.game_time(info))}, in level, alive")

    print('2. death on the frame it happens')
    for _ in range(60):
        C.emu_frame(env, C.ACTION_BUTTONS[2])
    # what Kill_Character does: routine 6, thrown up at -$700
    env.data.set_variable('t_yvel', {'address': 0xFFB01A, 'type': '>i2'})
    env.data.set_value('t_yvel', -0x700)
    env.data.set_value('mg_routine', 6)
    _, info = C.emu_frame(env, C.ACTION_BUTTONS[0])
    check(C.frame_flags(info)[2] and int(info['lives']) == 3,
          f"dead={C.frame_flags(info)[2]} with lives still {info['lives']}")
    n = 0
    while int(info['lives']) == 3 and n < 400:
        _, info = C.emu_frame(env, C.ACTION_BUTTONS[0])
        n += 1
    check(n > 30, f'the lives counter only drops {n + 1} frames later')

    print('3. down + left/right')
    rolled = {}
    for name, mask in (('down', C.ACTION_BUTTONS[3]),
                       ('down+right', [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])):
        env.reset()
        env.em.set_state(C.integration_state(C.DEFAULT_START))
        for _ in range(90):
            C.emu_frame(env, C.ACTION_BUTTONS[2])
        for _ in range(10):
            _, info = C.emu_frame(env, mask)
        rolled[name] = bool(int(info['mg_status']) & 4)
    check(rolled == {'down': True, 'down+right': False},
          f'rolling at speed: {rolled} (the game ignores down while left/right is held)')
    k = record_play.keys_to_action
    table = {(0, 0, 0, 0, 0, 0): 0, (1, 0, 0, 0, 0, 0): 1, (0, 1, 0, 0, 0, 0): 2,
             (0, 0, 0, 1, 0, 0): 3, (0, 0, 1, 0, 0, 0): 4, (0, 0, 0, 0, 1, 0): 5,
             (1, 0, 0, 0, 1, 0): 6, (0, 1, 0, 0, 1, 0): 7, (0, 0, 0, 1, 1, 0): 8,
             (0, 1, 0, 1, 0, 0): 2, (0, 1, 0, 1, 1, 0): 7, (1, 1, 0, 0, 0, 0): 0,
             (0, 1, 0, 1, 1, 1): 8, (0, 1, 0, 1, 0, 1): 3, (0, 0, 1, 0, 1, 0): 5}
    bad = {key: (k(*key), v) for key, v in table.items() if k(*key) != v}
    check(not bad, f'keys_to_action on {len(table)} combinations' + (f': {bad}' if bad else ''))
    env.close()


def mini_act():
    print('4. a scripted mini-act, cleared by a test signpost')
    env = C.make_retro()
    env.reset()
    start = C.integration_state(C.DEFAULT_START)
    env.em.set_state(start)
    split = C.ActSplitter()
    actions, times = [], []
    i = 0
    while not split.acts or i <= split.acts[-1]['clear'] + 60:
        _, info = C.emu_frame(env, C.ACTION_BUTTONS[script_action(i)])
        split.feed(i, info)
        if split.acts:
            times.append(C.game_time(info))
        actions.append(script_action(i))
        i += 1
        if i > 4000:
            break
    env.close()
    check(len(split.acts) == 1, f'acts: {split.summary()}')
    if not split.acts:
        return
    act = split.acts[0]
    check(len(set(times[:30])) == 1, f"{act['name']} cleared at frame {act['clear']}; the "
          f"timer holds {act['time']} afterwards")
    demos.save_play(SYNTH, actions, dict(
        start_state=C.DEFAULT_START, start_sha1=C.sha1(start),
        action_table=C.ACTION_BUTTONS, action_names=C.ACTION_NAMES, frames=len(actions),
        acts=split.summary(), rewinds=0, sessions=[],
        synthetic='MG_TEST_GOAL_X=7100: scripted run right, jump the ledge'))
    ok, note = demos.replay_play_check(SYNTH)
    check(ok, note)


def invariant():
    print('5. the demo\'s own inputs from every start point')
    actions, _ = demos.load_play(SYNTH)
    env = C.SonicCoordinator(demo_files=[SYNTH], acts=['AIZ1'])
    env.demo_info()
    seg = env._segs[0]
    worst = []
    for k, ck in enumerate(seg['ckpts']):
        obs, _ = env.reset(options={'key': seg['key'], 'ckpt': k})
        f, m, total = ck['frame'], 0, 0.0
        stack_ok = np.array_equal(obs, ck['stack'])
        while True:
            j = f + 1 + 4 * m
            _, r, term, trunc, info = env.step(int(actions[j]) if j < len(actions) else 0)
            total += r
            m += 1
            if term or trunc:
                break
        res = info['mg_result']
        ret = total / C.REWARD_SCALE
        if not (stack_ok and res['outcome'] == 'clear' and res['frames'] == res['target']
                and abs(ret - C.CLEAR_BONUS) < 1e-6):
            worst.append((k, res['outcome'], res['frames'], res['target'], round(ret, 2)))
    check(not worst, f"{len(seg['ckpts'])} start points: clear on the demo's frame, return "
          f"= CLEAR_BONUS {C.CLEAR_BONUS:.0f}" + (f'; MISMATCHES {worst}' if worst else ''))
    env.reset(options={'key': seg['key'], 'ckpt': 3})
    while True:                                     # stand still: no route progress
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    res = info['mg_result']
    check(res['outcome'] == 'stuck' and C.STUCK_FRAMES <= res['frames'] <= C.STUCK_FRAMES + 200,
          f"standing still is cut as '{res['outcome']}' after {res['frames']} frames "
          f"(STUCK_FRAMES {C.STUCK_FRAMES}; the timeout would be {env._max_frames})")
    env.close()


def recorder():
    print('6. the recorder, headless')
    path = os.path.join(demos.DEMO_DIR, '_synthetic_recorder_check.npz')
    rec = record_play.PlayRecorder(path=path)
    for i in range(300):
        rec.step(script_action(i))
    rec.rewind()                                    # ~2 s back
    back = len(rec.actions)
    for i in range(back, 700):
        rec.step(script_action(i))
    rec.save()
    rec.close()
    ok, note = demos.replay_play_check(path)
    check(ok and back == 180, f'rewound 300 -> {back}; {note}')
    rec = record_play.PlayRecorder(resume=path)
    cached = rec.from_cache                         # restored from the cache, not replayed
    rec.rewind()                                    # back past where the last session ended
    rewound = len(rec.actions)
    for i in range(rewound, 760):
        rec.step(script_action(i))
    rec.save()
    rec.close()
    ok, note = demos.replay_play_check(path)
    acts = demos.load_play(path)[1]['acts']
    check(ok and cached and rewound == 580 and len(demos.load_play(path)[0]) == 760,
          f'resumed from the cache={cached}, rewound past the resume point to {rewound}, '
          f'extended to 760 frames; {note}')
    check([a['clear'] for a in acts] == [a['clear'] for a in demos.load_play(SYNTH)[1]['acts']],
          'the recorder found the same clear frame as the scripted run')
    for p in (path, os.path.join(record_play.CACHE_DIR, demos.demo_id(path) + '.resume')):
        if os.path.exists(p):
            os.remove(p)


def playthroughs():
    print('7. recorded playthroughs')
    C.TEST_GOAL_X = 0                               # the fixture is for the mini-act only
    files = sorted(glob.glob(os.path.join(demos.DEMO_DIR, 'play_*.npz')))
    if not files:
        print('  (none recorded yet — python3 record_play.py)')
    for p in files:
        ok, note = demos.replay_play_check(p)
        check(ok, f'{os.path.basename(p)}: {note}')


if __name__ == '__main__':
    ram_map()
    mini_act()
    invariant()
    recorder()
    playthroughs()
    print('\nALL CHECKS PASSED' if not failures else f'\n{len(failures)} FAILED: {failures}')
    sys.exit(1 if failures else 0)
