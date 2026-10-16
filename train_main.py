"""
train_main.py — the base-game trainer: PPO on coordinator.SonicCoordinator,
learning each act from Tyler's recorded playthrough and then beating it.

Each run is a named entry in RUNS: which acts, which recordings, the
hyperparameters, and the result. A run is data, not a file (see the
consolidation notes in the context doc for why).

LAUNCH:
  cd ~/Projects/sonic-rl && source venv/bin/activate
  python3 train_main.py --list
  python3 smoke_main.py                         # env + pipeline checks
  nohup python3 train_main.py aiz1 > logs/main_aiz1.log 2>&1 &
  grep -E "FRESH|Resuming" logs/main_aiz1.log   # MUST say FRESH for a new run
  python3 main_progress.py --run aiz1           # catching up with you? beating you?

HOW IT LEARNS FROM YOUR PLAY — the backward curriculum of Salimans & Chen
(2018), per act. Episodes start at a start point along your recording (one
per second of it). The start point begins near the END of the act and moves
one second back whenever >= 25% of the recent episodes from it matched you:
cleared the act from there within 1.25x your time + 1 s. At the first start
point the agent is playing the act from its beginning. A share of episodes
(1 - demo_prob) always starts at the beginning, as the honest measure.

The agent never copies your inputs. The reward is "frames faster than you,
along your own route" (see coordinator.py), so once it matches you it keeps
being paid for every frame it saves, including by finding a route you didn't
take.

RESUMING an existing run requires --resume; without it the launcher refuses
when a checkpoint is already on disk (the v2 contamination lesson).

STOPPING (best first):
  touch ~/Projects/sonic-rl/STOP_main_<run>    # saves, then exits
  Ctrl-C / kill <parent PID>                   # signal handler saves
Autosaves every AUTOSAVE_STEPS regardless.
"""

import os
import sys
import glob
import json
import shutil
import signal
import argparse
import collections
import time

import numpy as np
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                CallbackList)

# ── Run registry ─────────────────────────────────────────────────────────────
# `acts` are act names as coordinator.act_name / part_name make them ('AIZ1', 'LRZ3', 'LBZ2b'...);
# `demos` are patterns under demos/.
RUNS = {
    'smoke': dict(
        acts=['AIZ1'], demos=['_synthetic_aiz1_goal7100'], demo_prob=0.75,
        test_goal_x=7100, lr=2.5e-4, steps=3_000_000,
        result='Pipeline check, not an experiment: a scripted 11 s "act" (AIZ1 '
               'from its start to a test signpost at x=7100, made by '
               'smoke_main.py). Expect the start point to walk back to 0 and '
               'the agent to beat the script (it wastes ~0.7 s on a wall).',
    ),
    'aiz1': dict(
        acts=['AIZ1'],
        # pinned 2026-10-10: 'play_*' would pull Tyler's coming full-game
        # recording into a resume
        demos=['play_20260930-153946'], demo_prob=0.75,
        lr=2.5e-4, steps=50_000_000,
        result="LAUNCHED 2026-09-30 on play_20260930-153946 (Tyler, AIZ1 1:25.43, "
               'no deaths or rewinds; 84 start points; fights = the jungle-burning '
               'flyover and the miniboss). The questions: does the start point reach '
               'the beginning of the act, and how many frames does it beat him by?',
    ),
    'mg1': dict(
        acts=['AIZ1'], demos=['play_20260930-153946'], demo_prob=0.75,
        lr=2.5e-4, steps=50_000_000, env_version=2,
        ss='pilot',     # from 2026-10-12 ~09:10 (Tyler: "can we have special stages turned on?")
        # from 2026-10-12 ~10:05 at 4,459,616 steps (Tyler: "Let's go with your suggestion"):
        # the pointer moved back to 72 (just before AIZ1's wall climb) and judged only on
        # episodes starting AT it, a full window of 32 (>= 25% matched) before it moves
        curr_at_pointer=True, curr_min_n=32,
        # from 2026-10-12 ~10:40 (Tyler: "I don't want to discourage it from finding big
        # rings ... I'd like the agent to get all 14 emeralds and beat the secret final stage"):
        # a won stage's trip is free and each emerald pays 300 frames (5 s) on top
        ss_detour='refund', emerald_bonus=300,
        # from 2026-10-12 ~11:20 (Tyler: "Go ahead"): stepping stones up AIZ1's climb (rock
        # pile t 4,320 -> collapsing ledge ~4,362 -> top ~4,392): take-off, mid-jump, on the
        # ledge; start points 0-72 keep their numbers, the new ones are 73-75
        extra_starts={'AIZ1': [4335, 4350, 4365]},
        result='STARTED 2026-10-12 ~07:40 (Tyler: "i want to start working on the main game"; '
               'd10 paused, board 14 back overnight). Criteria still Claude\'s proposal (doc, '
               'Pre-Committed Criteria "Run mg1"). aiz1 exactly, but on MG env v2 (new ground '
               'counts against stuck; off-route time settled; route ties by the clock): '
               'does v2 get past the wall that stopped aiz1 at start point 69 (82%)?',
    ),
    'mg1bc': dict(
        # mg1 exactly as it runs now, but from the imitation-pretrained network (train_bc.py bc1:
        # mg1's network taught Tyler's actions on 407k frames of his play, 52% held-out accuracy)
        acts=['AIZ1'], demos=['play_20260930-153946'], demo_prob=0.75,
        lr=2.5e-4, steps=50_000_000, env_version=2,
        init='bc1', ss='pilot', ss_detour='refund', emerald_bonus=300,
        curr_at_pointer=True, curr_min_n=32,
        # + 4,095 / 4,110 / 4,125 from ~12:15 (the stepping-stones flag at start point 68:
        # 0 of 32 matched, 87% stuck at x 3,264; his route loops up-left through a path
        # swapper at (3200, 960) there)
        extra_starts={'AIZ1': [4095, 4110, 4125, 4335, 4350, 4365]},
        # from its resume at 2026-10-12 ~13:24 (Tyler: "yes please. go ahead and build that"):
        # a fading pull toward his inputs, half of it at the start point it's stuck at
        imitation=dict(data='datasets/bc_v1'),
        result='STARTED 2026-10-12 11:27 (Tyler: "This sounds great!" to imitation first, then '
               '"you can have the box"). Launched at mg1\'s start point 75 (on the collapsing ledge; '
               'curriculum file copied from mg1 at 6,874,240 steps). Pretrained alone, 10 sampled '
               'tries each: from 75 bc1 clears 4/10 near his pace, mg1 0/10; from 72 both 0/10. '
               'Reads (Claude\'s, proposed): off 75 within 1M steps (mg1: 0 of 32 matched there in '
               '~0.9M); past 72 within 5M; forgetting = its match rate at 75 falling from ~40% '
               'toward 0 early (then add a decaying imitation loss). RESULT 2026-10-12 ~13:35: '
               'past the wall in 17 min; with imitation from ~13:24, through the loop-back; '
               'FIRST FRESH CLEARS of AIZ1 at ~4.5M steps, best 1:14.65 vs his 1:25.43.',
    ),
    'mg2': dict(
        # AIZ1 exactly as mg1 trains it (same recording, start points, stepping stones);
        # the next five acts from his whole-game recording
        acts=['play_20260930-153946:AIZ1'] + [f'play_20261010-202923:{a}' for a in
                                              ('AIZ2', 'HCZ1', 'HCZ2', 'MGZ1', 'MGZ2')],
        demos=['play_20260930-153946', 'play_20261010-202923'], demo_prob=0.75,
        lr=2.5e-4, steps=100_000_000, env_version=2,
        # from mg1bc's own network (it clears AIZ1 from the start: 5/20), no upfront copying
        # pass: one (bc4; even bc5 with AIZ1 left out) dropped that to 0/20; the fading
        # imitation pull teaches the five new acts instead. mg1bc's curriculum file copied at
        # launch (AIZ1 starts at 63). First attempt from bc4: logs/archive_mg2_from_bc4/
        init='mg1bc', air_bar=True,
        ss='pilot', ss_detour='refund', emerald_bonus=3600,   # 300, then 1200 (emerald_pay below), then 3600 (ring_share below)
        curr_at_pointer=True, curr_min_n=32, curr_successes=8,
        extra_starts={'play_20260930-153946:AIZ1': [4095, 4110, 4125, 4335, 4350, 4365]},
        # from its resume after 2026-10-12 ~14:25 (Tyler: "Let's try the compass"): the ring
        # compass on, and the imitation frames drawn with it (bc_v3 = bar + compass)
        ring_compass=True,
        # floor 0.3 from ~19:30 (the pull reaching 0 at 3M steps collapsed every act: rolled
        # back to the 2,261,000-step checkpoint, logs/archive_mg2_collapse/)
        imitation=dict(data='datasets/bc_v4', floor=0.3, front=True),   # front: from ~60.3M (see fight_slack)
        # from its resume at 6,241,416 steps (2026-10-14 ~20:55; Tyler: "let's add 2 and 3 as well",
        # after the boss hit counter was found): the boss bar drawn in (bc_v4 = bar + compass + boss
        # bar; its imitation fade carried over from bc_v3's, so the pull stays at the floor), each
        # hit paid 120 frames up front and taken back with the fight's lump, and a hit counts as
        # progress for `stuck` (coordinator "Boss hits"; check_boss.py)
        boss_bar=True, boss_pay=120,
        # from its resume at ~36.9M steps (2026-10-15 ~06:50; Tyler: "Let's try your recommendation"):
        # each emerald pays 1,200 frames (20 s; was 300 at the stage) once Sonic makes route progress
        # again after the stage (or clears), nothing if he never gets back: it had learned to skip
        # rings (AIZ1 fresh starts with the ring cleared 28% vs 63%, stuck in its pocket; took it
        # 8% -> <1%); check_emerald_pay.py
        emerald_pay='return',
        # from its resume at ~50M steps (2026-10-15; Tyler: "lets go with the boss share inside
        # mg2"): AIZ2, HCZ2 and MGZ2 were parked inside their end-of-act boss fights while AIZ1,
        # cleared from the start (best 3,986 frames vs his 5,060), took 25% of the frames and AIZ2
        # 7%. A quarter of resets now start inside a fight an act is stuck in (fight start ..
        # pointer band), and AIZ1 is drawn at 0.3 the rate of the others for the rest (~6% of
        # the frames; the imitation pull keeps his AIZ1 frames too); check_boss_share.py
        # AIZ1 weight 0.3 -> 1.0 from its resume at ~64.2M steps (2026-10-15 ~22:40; Tyler: "Make the change"):
        # at 0.3, once the boss/ring/front shares and the time balance went in, AIZ1 got ~4% of the steps and
        # ~20-60 main episodes an hour (was ~1,000), and forgot: clears 60-74% (overnight) -> 14-20% from ~15:00,
        # stuck at his climb (x ~10,500-11,600) and the burnt part's wall
        boss_share=0.25, act_weights={'play_20260930-153946:AIZ1': 1.0},
        # from the same day's next resume (Tyler: "i think the goal is to encourage sonic to grab all
        # 14 giant rings. speed can come afterwards"; "let's try it out"): 15% of resets start where
        # his route passes nearest a giant ring (14 within 400 px in these acts), each emerald pays
        # 3,600 frames (60 s, two deaths; was 1,200) on getting back, and the curriculum's time bar
        # leaves out a won stage and its refunded trip; check_ring_share.py
        ring_share=0.15, bar_forgive=True,
        # from the next resume (Tyler: "yes, balance it by time"): episodes differ ~3x in length, so
        # by episode count HCZ2 + MGZ2 took 58% of the steps and AIZ2 10%. Now the shares are of
        # steps, each split evenly in steps (AIZ1 at 0.3); check_balance_time.py
        balance_time=True,
        # from its resume at ~58.2M steps (2026-10-15 ~17:30; Tyler: "yes" to building the clip training):
        # rings with a detour clip (ring_clips.py) start their ring-share episodes at a point along it (the
        # last 30 s before he went in), not on his route, where 43 of 44 probe tries never got closer;
        # check_ring_clips.py
        ring_clips='demos/clips',
        # from its resume at ~60.3M steps (2026-10-15 ~18:20; Tyler: "yes" to aiming the imitation pull at the
        # stuck openings, "and maybe we should loosen the boss fights"): AIZ2/HCZ2/MGZ2's pointers sat in their
        # fights all day (MGZ2 18M steps), clearing at ~1.4-1.7x his time; a start inside his fight now matches
        # him within 2.5x (MG_FIGHT_SLACK), as proposed 13:20 and declined then; and imitation 'front' adds his
        # frames at each act's fresh-start frontier (HCZ2: frames 0-581, his spin dash and roll up the curve)
        fight_slack=1.5,
        # from its next resume (2026-10-16; Tyler: "Yes that sounds great" to a passive main-game route book):
        # fresh clears faster than an act's book entry are saved for main_book.py harvest; training unchanged
        book='routes_main',
        # from its next resume (2026-10-16; Tyler: "You can turn those both on"): MGZ1's end boss (no Boss_flag)
        # becomes a fight by its object code; check_fight_codes.py
        fight_codes=True,
        # from its resume at ~56.6M steps (2026-10-15 ~15:30; Tyler, after HCZ2's opening was measured:
        # "Yes, it's probably a good use of time"): its fresh starts all stopped at 3.4% (a curve that must
        # be rolled up; the agent hops in the corridor and lands running), then at a sliding wall, and the
        # curriculum, at 81%, wouldn't reach the opening for a long time. 15% of the steps start in the
        # 10 s of his route up to each act's fresh-start frontier (the median of its last 32 fresh starts),
        # practice following the frontier as it moves; check_front_share.py
        front_share=0.15,
        result='STARTED 2026-10-12 ~13:50 from bc4, criteria AGREED (Tyler: "let\'s do it!" after '
               'mg1bc cleared AIZ1 from the start); RESTARTED ~16:25 from mg1bc (bc4 had lost AIZ1 '
               'from the start: 0/20 vs mg1bc 5/20). Six acts at once: do the five new acts learn, '
               'and does sharing hurt AIZ1? (doc "Run mg2")',
    ),
}

# ── Shared config ────────────────────────────────────────────────────────────
NUM_ENVS       = 8
AUTOSAVE_STEPS = 100_000
CKPT_EVERY     = 500_000       # env steps between rolling checkpoints
KEEP_CKPTS     = 5
HOME           = os.path.dirname(os.path.abspath(__file__))

PPO_KWARGS = dict(
    n_steps=512, batch_size=512, n_epochs=4,
    gamma=0.995,               # 4-frame steps: a ~13 s horizon
    gae_lambda=0.95, clip_range=0.1, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
)

# Imitation during RL (RUNS 'imitation': dict(data=..., and any of these to override))
IMITATION_DEFAULTS = dict(weight=1.0, decay_steps=3_000_000, steps=8, batch=256,
                          behind=120, ahead=600, floor=0.0)

# Backward curriculum
CURR_WINDOW  = 32      # recent episodes judged at the current start point
CURR_MIN_N   = 16      # ...needed before it may move
CURR_ADVANCE = 0.25    # move one start point back once >= 25% of them matched


def paths(run):
    return dict(
        zip=os.path.join(HOME, f'main_{run}'),
        pkl=os.path.join(HOME, f'main_{run}_vecnormalize.pkl'),
        ckpt_dir=os.path.join(HOME, f'checkpoints_main_{run}'),
        stop=os.path.join(HOME, f'STOP_main_{run}'),
        log=os.path.join(HOME, f'mg_log_{run}.jsonl'),
        curriculum=os.path.join(HOME, f'main_{run}_curriculum.json'),
        stones=os.path.join(HOME, f'main_{run}_stones.json'),     # automatic stepping stones
        starts=os.path.join(HOME, f'main_{run}_starts.json'),     # every act's start-point times
        prefix=f'main_{run}',
    )


def demo_paths(cfg):
    found = []
    for pat in cfg['demos']:
        found += sorted(glob.glob(os.path.join(HOME, 'demos', f'{pat}.npz')))
    return found


def list_runs():
    for name, cfg in RUNS.items():
        p = paths(name)
        status = (f"checkpoint on disk ({os.path.getsize(p['zip'] + '.zip') / 1e6:.0f} MB)"
                  if os.path.exists(p['zip'] + '.zip') else 'no checkpoint')
        print(f"{name:6} acts {','.join(cfg['acts']):10} demos {cfg['demos']} "
              f"({len(demo_paths(cfg))} found)  lr {cfg['lr']:.1e}  {status}")
        print(f"       {cfg['result']}")


def extra_starts(run):
    """{act key: sorted demo frames} of a run's extra start points: RUNS 'extra_starts' and the
    stepping stones mg_stones.py adds when a start point stalls (main_<run>_stones.json)."""
    out = {k: set(v) for k, v in RUNS[run].get('extra_starts', {}).items()}
    try:
        with open(paths(run)['stones']) as f:
            for k, v in json.load(f).items():
                out.setdefault(k, set()).update(v)
    except (OSError, ValueError):
        pass
    return {k: sorted(v) for k, v in out.items()}


def ring_clip_files(where):
    """The clip files (ring_clips.py) under directory `where` that went into their ring, sorted; [] if none."""
    if not where:
        return []
    import demos
    out = []
    for p in sorted(glob.glob(os.path.join(HOME, where, '*.npz'))):
        try:
            if (demos.load_play(p)[1].get('clip') or {}).get('entered_at') is not None:
                out.append(p)
        except ValueError:
            pass
    return out


def configure_env(run):
    """Point coordinator.py at a run's acts and recordings. Must happen before
    anything imports coordinator (it reads MG_* at import, and SubprocVecEnv
    workers inherit this environment). Returns the demo files."""
    cfg, p = RUNS[run], paths(run)
    found = demo_paths(cfg)
    os.environ['MG_DEMOS'] = ','.join(found)
    os.environ['MG_ACTS'] = ','.join(cfg['acts'])
    os.environ['MG_DEMO_PROB'] = str(cfg['demo_prob'])
    os.environ['MG_LOG'] = p['log']
    os.environ['MG_TEST_GOAL_X'] = str(cfg.get('test_goal_x', 0))
    # 1 = the 2026-09-30 env (smoke, aiz1); new runs set env_version=2
    # (new ground counts against `stuck`; off-route time settled on return)
    os.environ['MG_ENV_VERSION'] = str(cfg.get('env_version', 1))
    # special stages: 'truncate' (default: a giant ring ends the episode) or
    # 'pilot' (the special-stage agent / route book plays it; coordinator.py)
    os.environ['MG_SS'] = cfg.get('ss', 'truncate')
    # with 'pilot': frames per emerald won, on top of the refund; 'refund' = a won
    # stage's trip to the ring and back is free too (coordinator.py)
    os.environ['MG_EMERALD_BONUS'] = str(cfg.get('emerald_bonus', 0))
    # when: 'stage' (at the special stage) or 'return' (once Sonic makes route progress again)
    os.environ['MG_EMERALD_PAY'] = cfg.get('emerald_pay', 'stage')
    os.environ['MG_SS_DETOUR'] = cfg.get('ss_detour', 'charge')
    # Sonic's air drawn into the observation (coordinator.obs_frame)
    os.environ['MG_AIR_BAR'] = '1' if cfg.get('air_bar') else '0'
    # the nearest unused giant ring drawn into the observation (coordinator.draw_compass)
    os.environ['MG_RING_COMPASS'] = '1' if cfg.get('ring_compass') else '0'
    # the boss's hits left drawn into the observation, each hit paid up front (frames, taken
    # back with the fight's lump); either one also lets a hit count as progress for `stuck`
    # (coordinator, "Boss hits")
    os.environ['MG_BOSS_BAR'] = '1' if cfg.get('boss_bar') else '0'
    os.environ['MG_BOSS_PAY'] = str(cfg.get('boss_pay', 0))
    # a share of resets started inside the boss fight an act's curriculum is stuck in, and how
    # often each act is drawn for the rest (coordinator, "Boss share")
    os.environ['MG_BOSS_SHARE'] = str(cfg.get('boss_share', 0))
    os.environ['MG_ACT_WEIGHTS'] = ';'.join(f'{k}:{w}' for k, w in cfg.get('act_weights', {}).items())
    # a share of resets started where his route passes nearest a giant ring, and the time bar
    # leaving out what the reward forgives (coordinator, "Ring share")
    os.environ['MG_RING_SHARE'] = str(cfg.get('ring_share', 0))
    os.environ['MG_BAR_FORGIVE'] = '1' if cfg.get('bar_forgive') else '0'
    # a share of resets started just before where an act's fresh starts get stuck, so its opening
    # is practised from his states while the curriculum is still at its end (coordinator, "Front share")
    os.environ['MG_FRONT_SHARE'] = str(cfg.get('front_share', 0))
    # a looser match bar for starts inside his boss fights (coordinator, "Fight bar")
    os.environ['MG_FIGHT_SLACK'] = str(cfg.get('fight_slack', 0))
    # fights recognized by the boss's object code too, not only Boss_flag (coordinator, "Fights by object code")
    os.environ['MG_FIGHT_CODES'] = '1' if cfg.get('fight_codes') else '0'
    # the passive main-game route book: fresh clears faster than an act's best are kept for main_book.py
    # (coordinator, "Main-game route book")
    os.environ['MG_BOOK'] = os.path.join(HOME, cfg['book']) if cfg.get('book') else ''
    # ring-share starts along Tyler's detour clips into giant rings (coordinator, "Ring clips"): the clips
    # in RUNS 'ring_clips' (a directory) that went into their ring, so a resume picks up new ones
    os.environ['MG_RING_CLIPS'] = ','.join(ring_clip_files(cfg.get('ring_clips')))
    # the shares and act weights as shares of training time (steps), not episodes (coordinator,
    # "Time balance")
    os.environ['MG_BALANCE_TIME'] = '1' if cfg.get('balance_time') else '0'
    # extra start points: the RUNS entry's plus the stepping stones mg_stones.py added
    # automatically when a start point stalled (main_<run>_stones.json)
    os.environ['MG_EXTRA_STARTS'] = ';'.join(f"{a}:{','.join(map(str, ts))}"
                                             for a, ts in extra_starts(run).items())
    return found


_STOP = {'requested': False, 'why': ''}
P = None                        # the run's paths(), set by main()


def _request_stop(signum, frame):
    _STOP.update(requested=True, why=f'signal {signum}')
    print(f'\n[GracefulStop] caught signal {signum} — will save and exit at the '
          f'next step boundary.', flush=True)


def save_all(model, venv, tag, steps):
    try:
        model.save(P['zip'])
        venv.save(P['pkl'])
        print(f"[{tag}] saved at {steps} steps -> {P['zip']}.zip + {P['pkl']}", flush=True)
        return True
    except Exception as e:
        print(f'[{tag}] SAVE FAILED at {steps} steps: {e!r}', flush=True)
        return False


class AutoSaveCallback(BaseCallback):
    def __init__(self, every=AUTOSAVE_STEPS):
        super().__init__()
        self.every, self.last = every, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            save_all(self.model, self.training_env, 'AutoSave', self.num_timesteps)
        return True


class GracefulStopCallback(BaseCallback):
    """Stop on a signal or the STOP file: save, then return False."""
    def __init__(self, stop_file, poll_every=200):
        super().__init__()
        self.stop_file, self.poll_every, self._last = stop_file, poll_every, 0

    def _on_step(self):
        if not _STOP['requested'] and self.num_timesteps - self._last >= self.poll_every:
            self._last = self.num_timesteps
            if os.path.exists(self.stop_file):
                _STOP.update(requested=True, why=f'stop file {self.stop_file}')
        if not _STOP['requested']:
            return True
        print(f"\n[GracefulStop] stopping ({_STOP['why']}) — saving now...", flush=True)
        save_all(self.model, self.training_env, 'GracefulStop', self.num_timesteps)
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        return False


class ImitationCallback(BaseCallback):
    """A fading pull toward Tyler's own inputs during RL (Tyler 2026-10-12:
    "yes please. go ahead and build that"). After each PPO update (at the
    start of the next rollout, so the rollout is collected by the policy it
    trains), `steps` extra gradient steps on the log-likelihood of his action
    on his frames (build_bc_dataset.py's pairs, weighted by how many of the 4
    frames agree), with its own Adam at PPO's learning rate times a weight
    that falls linearly from `weight` to 0 over `decay_steps` env steps from
    when it first ran on this data (kept in main_<run>_imitation.json, so a
    resume continues the fade instead of restarting it; new data restarts it). The weight scales the step size, not
    the loss: Adam undoes any constant scaling of a loss. Half of each batch comes
    from around each act's current curriculum start point (his frames from
    `behind` before it to `ahead` after it: the hard spot the pointer is
    stuck at), half from the run's whole acts (not to forget the rest). The
    value head is left alone. Logged under imitation/ (loss, weight,
    accuracy on the batch). With cfg 'front' (mg2, 2026-10-15; Tyler: "yes" to pointing it at HCZ2's
    opening, whose fresh starts all stop at a curve he rolls up and the agent hops into), the focused half
    also covers each act's fresh-start frontier: his frames in the front share's window, FRONT_BACK up to
    where the median fresh start gets (coordinator.fresh_frontier, re-read every FRONT_EVERY_S), for
    acts whose frontier is short of the pointer."""

    def __init__(self, cfg, run, curriculum_path):
        super().__init__()
        self.cfg, self.run, self.curriculum_path = dict(cfg), run, curriculum_path
        self.start = None
        self._front, self._front_at = None, 0.0     # cfg 'front': act key -> fresh-start frontier; when read

    def load_data(self):
        """His frames and samples for the run's acts (no model needed: check_imitation_front.py)."""
        import coordinator as C                     # imported by main() after configure_env
        self.C = C
        c = self.cfg
        data = os.path.join(HOME, c['data'])
        meta = json.load(open(os.path.join(data, 'meta.json')))
        S = dict(np.load(os.path.join(data, 'samples.npz')))
        self.frames = [np.load(os.path.join(data, f"frames_{d['id']}.npy"), mmap_mode='r')
                       for d in meta['demos']]
        demo_ids = [d['id'] for d in meta['demos']]
        trained_on = {os.path.basename(p)[:-4] for p in demo_paths(RUNS[self.run])}   # its recordings only
        keep = np.zeros(len(S['action']), dtype=bool)
        self.keys = {}                       # curriculum key -> sample indices of that act's recording
        for a in RUNS[self.run]['acts']:
            demo, _, name = a.rpartition(':')
            for di, did in enumerate(demo_ids):
                if (demo and did != demo) or did not in trained_on:
                    continue
                if name not in meta['acts']:
                    continue
                m = (S['demo'] == di) & (S['act'] == meta['acts'].index(name))
                if m.any():
                    keep |= m
                    self.keys[f'{did}:{name}'] = np.flatnonzero(m)
        self.S, self.all = S, np.flatnonzero(keep)
        self.offs = np.arange(-(C.STACK - 1) * C.FRAME_SKIP, 1, C.FRAME_SKIP)
        self.w = (S['agree'] / C.FRAME_SKIP).astype(np.float32)
        return meta

    def _on_training_start(self):
        import torch
        meta = self.load_data()
        c = self.cfg
        lr = self.model.learning_rate
        self.base_lr = lr(1.0) if callable(lr) else float(lr)
        self.opt = torch.optim.Adam(self.model.policy.parameters(), lr=self.base_lr)
        cp = self.curriculum_path
        state = (cp[:-len('_curriculum.json')] + '_imitation.json' if cp.endswith('_curriculum.json')
                 else cp + '.imitation.json')
        try:
            starts = json.load(open(state))
        except (OSError, ValueError):
            starts = {}
        self.start = int(starts.get(c['data'], self.model.num_timesteps))
        starts[c['data']] = self.start
        with open(state + '.tmp', 'w') as f:
            json.dump(starts, f)
        os.replace(state + '.tmp', state)
        for flag, env in (('air_bar', 'MG_AIR_BAR'), ('ring_compass', 'MG_RING_COMPASS'),
                          ('boss_bar', 'MG_BOSS_BAR')):
            if bool(meta.get(flag)) != (os.environ.get(env) == '1'):
                raise ValueError(f"imitation data {c['data']} {flag}={meta.get(flag)} but the run's "
                                 f"{env}={os.environ.get(env)}: the frames would differ")
        print(f"[imitation] {len(self.all):,} of his frames in {len(self.keys)} acts ({c['data']}); "
              f"weight {c['weight']} -> {c.get('floor', 0.0)} over {c['decay_steps']:,} steps from {self.start:,}; "
              f"{c['steps']} steps of {c['batch']} per update"
              + ("; the focused half also at each act's fresh-start frontier" if c.get('front') else ''), flush=True)

    def weight(self):
        """Falls linearly to `floor` (default 0) over decay_steps. mg2 (2026-10-12) showed why a
        floor: once the pull reached 0 at 3M steps, PPO's entropy bonus flattened the policy
        (entropy 0.6 -> 1.8 of a max 2.2) and every act's clears collapsed (AIZ1 41% -> 0%)."""
        done = (self.model.num_timesteps - self.start) / self.cfg['decay_steps']
        return max(self.cfg.get('floor', 0.0), self.cfg['weight'] * max(0.0, 1.0 - done))

    def frontier(self):
        """Sample indices near each act's current start point."""
        try:
            with open(self.curriculum_path) as f:
                cur = json.load(f)
        except (OSError, ValueError):
            return self.all
        out = []
        for key, idx in self.keys.items():
            if key not in cur:
                continue
            T = int(self.S['T'][idx[0]])
            t0 = cur[key]['frac'] * T
            t = self.S['t'][idx]
            near = idx[(t >= t0 - self.cfg['behind']) & (t <= t0 + self.cfg['ahead'])]
            out.append(near)
        if self.cfg.get('front'):
            out += self.front_windows(cur)
        out = np.concatenate(out) if out else np.array([], dtype=int)
        return out if len(out) else self.all

    def front_windows(self, cur):
        """cfg 'front': [sample indices] of his frames in each act's front-share window (the FRONT_BACK
        frames up to its fresh-start frontier), for acts whose frontier is short of the pointer."""
        C = self.C
        if self._front is None or time.time() - self._front_at >= C.FRONT_EVERY_S:
            by = collections.defaultdict(list)
            for r in C.log_tail(C.FRONT_TAIL_BYTES):
                by[f"{r.get('demo')}:{r.get('act')}"].append(r)
            self._front = {k: C.fresh_frontier(by[k]) for k in self.keys}
            self._front_at = time.time()
        out = []
        for key, idx in self.keys.items():
            f = self._front.get(key)
            if f is None or key not in cur:
                continue
            if f >= cur[key]['frac'] * int(self.S['T'][idx[0]]):
                continue                                # it reaches the pointer: the pointer window covers it
            t = self.S['t'][idx]
            out.append(idx[(t >= f - C.FRONT_BACK) & (t <= f)])
        return out

    def _on_rollout_start(self):
        import torch
        if self.start is None:
            return
        lam = self.weight()
        if lam <= 0:
            return
        c, pol = self.cfg, self.model.policy
        for g in self.opt.param_groups:
            g['lr'] = self.base_lr * lam                # the fade: smaller steps, same loss
        front, rng = self.frontier(), np.random.default_rng()
        tot = hits = n = 0.0
        for _ in range(c['steps']):
            half = c['batch'] // 2
            idx = np.concatenate([rng.choice(front, half), rng.choice(self.all, c['batch'] - half)])
            obs = np.empty((len(idx), self.C.STACK, self.C.OBS_SIZE, self.C.OBS_SIZE), dtype=np.uint8)
            for j, s in enumerate(idx):
                obs[j] = self.frames[self.S['demo'][s]][np.clip(self.S['frame'][s] + self.offs, 0, None)]
            obs_t = torch.as_tensor(obs, device=pol.device).float()
            act = torch.as_tensor(self.S['action'][idx], device=pol.device)
            w = torch.as_tensor(self.w[idx], device=pol.device)
            dist = pol.get_distribution(obs_t)
            logp = dist.log_prob(act)
            loss = -(w * logp).sum() / w.sum()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            self.opt.step()
            tot += loss.item()
            hits += (dist.distribution.probs.argmax(-1) == act).float().sum().item()
            n += len(idx)
        self.logger.record('imitation/loss', tot / c['steps'])
        self.logger.record('imitation/weight', lam)
        self.logger.record('imitation/accuracy', hits / n)
        self.logger.record('imitation/frontier_frames', len(front))

    def _on_step(self):
        return True


class PruneCheckpointsCallback(BaseCallback):
    def __init__(self, ckpt_dir, keep=KEEP_CKPTS, every=CKPT_EVERY):
        super().__init__()
        self.ckpt_dir, self.keep, self.every, self.last = ckpt_dir, keep, every, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            files = sorted(glob.glob(os.path.join(self.ckpt_dir, '*.zip')),
                           key=os.path.getmtime)
            for f in files[:-self.keep]:
                os.remove(f)
        return True


class ActCurriculumCallback(BaseCallback):
    """The backward curriculum, one pointer per act of each recording. A
    pointer starts at the act's last start point and moves one back (~1 s of
    the recording) when >= CURR_ADVANCE of the recent episodes from it
    matched the recording (coordinator: cleared within MATCH_SLACK of its
    time). Pointers persist next to the checkpoint, so --resume keeps them."""

    def __init__(self, state_path, band, at_pointer=False, min_n=CURR_MIN_N, successes=None):
        """at_pointer: judge only episodes that began AT the start point, and
        only on a full window of min_n of them. Starts are drawn from [pointer,
        pointer + band]; counting the whole band let the easier later starts
        (past a hard spot) move the pointer before the hard spot itself was
        learned (mg1 at AIZ1's wall, 2026-10-12). successes: instead, move as
        soon as this many of the recent (CURR_WINDOW) episodes matched: 8 of
        32 is the same 25% bar where it's marginal, but an easy start point
        moves after 8 (mg2: six acts share the training)."""
        super().__init__()
        self.state_path, self.band = state_path, band
        self.at_pointer, self.min_n, self.successes = at_pointer, min_n, successes
        self.ptr, self.fracs, self.recent = {}, {}, {}

    def _on_training_start(self):
        saved = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                saved = json.load(f)
        infos = self.training_env.env_method('demo_info', indices=[0])[0]
        starts = {d['key']: [round(f * d['T']) for f in d['fracs']] for d in infos}
        tmp = self.state_path.replace('_curriculum.json', '_starts.json')
        if tmp != self.state_path:                  # demo times of every start point, for mg_stones.py
            with open(tmp + '.tmp', 'w') as f:
                json.dump(starts, f)
            os.replace(tmp + '.tmp', tmp)
        for d in infos:
            key, last = d['key'], len(d['fracs']) - 1
            self.fracs[key] = d['fracs']
            self.ptr[key] = min(saved.get(key, {}).get('ptr', last), last)
            self.recent[key] = collections.deque(maxlen=CURR_WINDOW)
            print(f"[curriculum] {key}: your time {d['time']} ({d['T']} frames), "
                  f"{last + 1} start points; starting {d['fracs'][self.ptr[key]]:.0%} "
                  f"into the act", flush=True)
        self._push()

    def _push(self):
        self.training_env.env_method('set_demo_pointers', dict(self.ptr))
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({k: {'ptr': p, 'frac': round(self.fracs[k][p], 4)}
                       for k, p in self.ptr.items()}, f, indent=1)
        os.replace(tmp, self.state_path)

    def _on_step(self):
        moved = False
        for info in self.locals.get('infos', []):
            r = info.get('mg_result')
            if not r or r['key'] not in self.ptr or r.get('clip'):
                continue                        # a ring clip's start isn't on his route (coordinator MG_RING_CLIPS)
            p = self.ptr[r['key']]
            if not p <= r['start'] <= (p if self.at_pointer else p + self.band):
                continue                    # began before the pointer moved (or, at_pointer, past it)
            q = self.recent[r['key']]
            q.append(r)
            if p > 0 and (len(q) >= self.min_n if self.successes is None else True):
                rate = sum(x['success'] for x in q) / len(q)
                if (rate >= CURR_ADVANCE if self.successes is None
                        else sum(x['success'] for x in q) >= self.successes):
                    margins = [x['margin'] for x in q if x['margin'] is not None]
                    fr = self.fracs[r['key']]
                    self.ptr[r['key']] = p - 1
                    q.clear()
                    moved = True
                    print(f"[curriculum] {r['key']}: start -> {fr[p - 1]:.0%} into the "
                          f"act ({rate:.0%} matched you from {fr[p]:.0%}; clears "
                          f"averaged {np.mean(margins):+.0f} frames vs you)", flush=True)
        if moved:
            self._push()
        return True

    def _on_rollout_end(self):
        for key, p in self.ptr.items():
            self.logger.record(f'curriculum/{key}/start_frac', self.fracs[key][p])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?', help='run name from RUNS')
    ap.add_argument('--list', action='store_true', help='show runs and exit')
    ap.add_argument('--resume', action='store_true',
                    help='continue an existing checkpoint (required when one exists)')
    args = ap.parse_args()
    if args.list or not args.run:
        list_runs()
        sys.exit(0 if args.list else 2)
    if args.run not in RUNS:
        sys.exit(f'unknown run {args.run!r}. Known: {", ".join(RUNS)} '
                 f'(add an entry to RUNS rather than copying this file).')
    global P
    CFG, P = RUNS[args.run], paths(args.run)
    DEMOS = configure_env(args.run)
    if not DEMOS:
        sys.exit(f"\nrun {args.run} learns from {CFG['demos']}, but none are in "
                 f"{os.path.join(HOME, 'demos')}.\n"
                 + ("  make it:  python3 smoke_main.py\n" if CFG.get('test_goal_x') else
                    "  record one:  python3 record_play.py\n"))

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    import coordinator

    def make_env():
        return coordinator.SonicCoordinator()

    resuming = os.path.exists(f"{P['zip']}.zip")
    print(f'=== base-game run {args.run} ===')
    print(f"  {CFG['result']}")
    print(f"  acts         : {CFG['acts']}")
    print(f"  demos        : {', '.join(os.path.basename(d) for d in DEMOS)} "
          f"({CFG['demo_prob']:.0%} of episodes on the curriculum)")
    print(f"  actions      : {len(coordinator.ACTION_BUTTONS)} x {coordinator.FRAME_SKIP} frames"
          f"   obs: {coordinator.OBS_SIZE}x{coordinator.OBS_SIZE}x{coordinator.STACK}")
    print(f"  lr {CFG['lr']:.1e}  {PPO_KWARGS}")
    if CFG.get('test_goal_x'):
        print(f"  TEST FIXTURE : signpost at x={CFG['test_goal_x']} (MG_TEST_GOAL_X)")
    print(f"  log          : {P['log']}")
    print(f"  graceful stop: touch {P['stop']}")
    print(f'  CUDA available: {torch.cuda.is_available()}')

    if resuming and not args.resume:
        sys.exit(f"\nREFUSING TO LAUNCH: {P['zip']}.zip already exists, so this would "
                 f"RESUME, not start fresh.\n"
                 f"  - to continue it:      add --resume\n"
                 f"  - to restart it clean: mv {P['zip']}.zip {P['pkl']} "
                 f"{P['curriculum']} aside first\n"
                 f"  - for a NEW experiment: add a new entry to RUNS\n")
    if args.resume and not resuming:
        sys.exit(f"\n--resume passed but no {P['zip']}.zip exists.\n")

    # every recording must still replay into the run it recorded: the start points are built here
    # once (or taken from start_cache, built from these same inputs before), refusing if an act lands
    # on other frames than recorded; the workers then load them instead of replaying (2026-10-16:
    # this was a whole-recording check here plus a rebuild in every worker, ~6 min)
    try:
        note = coordinator.prebuild()
    except ValueError as e:
        sys.exit(f'\nREFUSING TO LAUNCH: the recordings no longer replay: {e}\n')
    print(f'  start points : {note}', flush=True)

    os.makedirs(P['ckpt_dir'], exist_ok=True)
    tb_dir = os.path.join(HOME, 'logs_main')
    os.makedirs(tb_dir, exist_ok=True)
    if os.path.exists(P['stop']):
        os.remove(P['stop'])

    init = paths(CFG['init']) if CFG.get('init') and not resuming else None
    if init and not os.path.exists(P['curriculum']) and os.path.exists(init['curriculum']):
        shutil.copy(init['curriculum'], P['curriculum'])   # its start points, for acts they share
        print(f"Start points from {CFG['init']}: {init['curriculum']}")
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    if os.path.exists(P['pkl']) or init:
        env = VecNormalize.load(P['pkl'] if os.path.exists(P['pkl']) else init['pkl'], env)
        env.training = True
        print(f"Resumed VecNormalize from {P['pkl'] if os.path.exists(P['pkl']) else init['pkl']}")
    else:
        # clip_reward is effectively off: returns are measured in frames, a
        # death is -18 and a clear +6 before normalization, and a clip of 10
        # silently capped the special-stage death penalty for months
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=1000.0, gamma=PPO_KWARGS['gamma'])

    if resuming:
        lr = os.environ.get('LR_OVERRIDE')
        model = PPO.load(P['zip'], env=env, device='auto',
                         custom_objects={'learning_rate': float(lr)} if lr else None)
        print(f'Resuming base-game {args.run} at {model.num_timesteps} steps | '
              f'gamma {model.gamma} lr {model.learning_rate}')
    elif init:
        model = PPO.load(init['zip'], env=env, device='auto', tensorboard_log=tb_dir,
                         custom_objects={'learning_rate': CFG['lr']})
        model.num_timesteps = 0                     # its own steps from 0
        print(f"Starting base-game {args.run} from {CFG['init']}'s weights "
              f"({init['zip']}.zip, trained {CFG['init']} steps not counted)")
    else:
        model = PPO('CnnPolicy', env, learning_rate=CFG['lr'], verbose=1,
                    tensorboard_log=tb_dir, device='auto', **PPO_KWARGS)
        print(f'Starting FRESH base-game {args.run} model (random weights)')

    callbacks = CallbackList([
        GracefulStopCallback(P['stop']),
        CheckpointCallback(save_freq=max(CKPT_EVERY // NUM_ENVS, 1),
                           save_path=P['ckpt_dir'], name_prefix=P['prefix']),
        AutoSaveCallback(),
        PruneCheckpointsCallback(P['ckpt_dir']),
        ActCurriculumCallback(P['curriculum'], coordinator.DEMO_BAND,
                              at_pointer=CFG.get('curr_at_pointer', False),
                              min_n=CFG.get('curr_min_n', CURR_MIN_N),
                              successes=CFG.get('curr_successes')),
    ] + ([ImitationCallback({**IMITATION_DEFAULTS, **CFG['imitation']}, args.run, P['curriculum'])]
         if CFG.get('imitation') else []))
    try:
        model.learn(total_timesteps=CFG['steps'], callback=callbacks,
                    reset_num_timesteps=False, tb_log_name=f'main_{args.run}')
    except BaseException as e:
        print(f'\n[exit] {type(e).__name__}: {e!r} — emergency save...', flush=True)
        save_all(model, env, 'EmergencySave', model.num_timesteps)
        raise
    finally:
        save_all(model, env, 'FinalSave', model.num_timesteps)
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
