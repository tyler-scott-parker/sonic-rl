"""
train_ss_multistage.py — the multistage special-stage trainer, all runs.

ONE trainer for every multistage run. Each run is a named entry in RUNS below;
the run name determines the checkpoint names, the log file, the stop sentinel,
the observation preset, and the hyperparameters being tested.

This replaces train_ss_multistage{,_v2,_v3,_v4,_v5}.py, which were sed-derived
copies of each other. That pattern had already produced live bugs — v5's own
docstring told you to launch `train_ss_multistage_v3.py` and to
`pkill -f train_ss_multistage_v3.py`, i.e. run the WRONG net against the WRONG
env while writing to v5's log. A run is data, not a file.

LAUNCH:
  cd ~/Projects/sonic-rl && source venv/bin/activate
  python3 train_ss_multistage.py --list            # show runs + status
  python3 smoke_stages.py v6                       # every board reads right?
  nohup python3 train_ss_multistage.py v6 > logs/ss_multistage_v6.log 2>&1 &
  nohup python3 train_ss_multistage.py v5 --resume > logs/ss_multistage_v5.log 2>&1 &
  grep -E "FRESH|Resuming|preset" logs/ss_multistage_v6.log
  #   ^ MUST say "Starting FRESH ... (random weights)" for a new experiment,
  #     and the preset line shows the env version (v6 = env=v2).

ENV VERSION: every run pins SS_ENV_VERSION in its preset. v1-v5 trained on
env v1, which had the logic errors found 2026-09-29 (grid blind to board rows
24-31, episodes ending ~670 frames after death, springs shown as red — see
special_stage_env.py); they must stay on v1. v6 is the first run on env v2.
Env v3 (d2 onward) adds 4-frame choices and the stall rule for demo learning.
Env v4 (d3 onward) makes it one choice per corner: check it with
`python3 smoke_corners.py d3` before launching. Env v5 (d4) adds
turn-then-jump: `python3 smoke_corners.py d4`.

LEARNING FROM YOUR PLAY: record a run with `python3 record_demo.py <stage>`
(slow motion + rewind; it lands in demos/). A RUNS entry with `demos=[...]`
starts a share of episodes (`demo_prob`) from states along those runs. The
start point begins near the END of each demo and walks back to the stage start
as the agent matches or beats the demo from it — the backward curriculum of
Salimans & Chen 2018 (DemoCurriculumCallback). The agent never copies your
inputs, only starts where you were, so it can end up better than the demo.
Progress: python3 demo_progress.py --run d1
`demos='book'` (d10) starts them along the route book's current best route
per board instead (route_book.py), switching live when the book improves;
check it with `python3 check_book_curriculum.py d10` before launching.

RESUMING an existing run requires an explicit --resume. Without it the launcher
REFUSES to start when a checkpoint for that name is already on disk — the exact
contamination that silently resumed a 1M-step partial during the v2 launch and
made the per-stage curve look like catastrophic interference.

STOPPING — three ways, best first:
  1. GRACEFUL (recommended, loses NOTHING):
         touch ~/Projects/sonic-rl/STOP_<run>
     The trainer notices within ~200 steps, saves model + VecNormalize, exits,
     then removes the sentinel so the next launch isn't instantly stopped.
  2. Ctrl-C / kill <parent PID>  -> signal handler saves, then exits.
  3. pkill -f train_ss_multistage -> WORKS but is the least safe: pkill also
     matches the SubprocVecEnv workers (they inherit the parent's cmdline), so
     workers may die before the parent finishes saving. Prefer method 1.

Autosaves every AUTOSAVE_STEPS (100K) regardless, so even a hard kill -9 costs
at most ~100K steps.

Override the learning rate on a resume with LR_OVERRIDE=1e-4 (goes through
custom_objects AND the live objects — a constructor edit is a no-op on resume).
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

ALL_14 = list(range(1, 15))

# ── Run registry ─────────────────────────────────────────────────────────────
# Every historical and future multistage run, as data. `obs` values become
# SS_* environment variables read by special_stage_env at import time.
#
# Result column is the point of keeping the dead runs here: it is the only
# place the hyperparameters and the outcome sit side by side.
RUNS = {
    'v1': dict(
        stages=[1, 2], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='1',
                 SS_USE_PIXELS='1', SS_PATCH_R='5', SS_DROP_POSITION='1'),
        result='DONE. Position-drop lifted transfer 0.06 -> 0.75 at 2 boards. '
               'NOT resumable: used the old 15-feat vector with 4 dead zeros.',
    ),
    'v2': dict(
        stages=[1, 2, 3, 4, 7], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='1',
                 SS_USE_PIXELS='1', SS_PATCH_R='5', SS_DROP_POSITION='1'),
        result='DONE, negative. Trained boards converged (~0.76 ratio) but '
               'zero-shot on held-out 5,6 = 0.09 -> MEMORIZED the 5-board set. '
               'NOT resumable (old 15-feat vector).',
    ),
    'v3': dict(
        stages=ALL_14, lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='1',
                 SS_USE_PIXELS='1', SS_PATCH_R='5', SS_DROP_POSITION='1'),
        result='DONE. Plateaued SHALLOW at ~3.0-3.5% mean %board. LR cool '
               '3e-4->1e-4 bought 3.0->3.5 then re-plateaued -> read as a '
               'capacity/representation ceiling, not an optimization one. '
               'NOT resumable (old 15-feat vector).',
    ),
    'v4': dict(
        stages=ALL_14, lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='1',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        result='DONE (~64M steps). Observation redesign: pixels dropped, patch '
               'widened 11x11->15x15, 4 dead RAM feats removed. Baseline to '
               "beat was v3's ~3.5% plateau. Env v1: blind to board rows "
               '24-31, ~43% of steps were post-death animation.',
    ),
    'v5': dict(
        stages=ALL_14, lstm=1024, lr=2e-4,
        obs=dict(SS_ENV_VERSION='1',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        result='Ran to ~19M steps (2026-07-30). 2x LSTM capacity vs v4 — THE '
               'variable under test. lr trimmed 3e-4->2e-4 for the bigger net. '
               'Env v1 (same bugs as v4); superseded by v6.',
    ),
    'v6': dict(
        stages=ALL_14, lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='2',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        result='NOT STARTED. First run on env v2 (2026-09-29 fixes: whole '
               'board visible, rounded patch centre, spring channel, episode '
               'ends on contact, a clear pays its emerald). Hyperparameters = '
               'v4 on purpose, so v6 vs v4 isolates the env fixes.',
    ),
    # Learning from Tyler's play. `demos` = patterns under demos/ (record with
    # record_demo.py); `demo_prob` = share of episodes that start on a demo.
    # `demo_mix` = share of demo starts drawn from any start point between the
    # curriculum pointer and the end instead of the band just after it
    # (SS_DEMO_MIX; default 0).
    # Pin a run's demos to exact files: a wildcard like 'stage1_*' would pull
    # a NEW recording into the run on its next resume and change the experiment.
    'd1': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='2',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        result='STOPPED 2026-09-30 at ~3.1M steps on Tyler\'s stage-1 clear: '
               '0 matches from its first start point (98/102) in ~2,500 '
               'episodes. Per-frame choices never held a straight line (53% '
               'of frames pressed a turn), and it learned to hop around the red '
               'board to postpone dying (median 752 frames alive; the demo '
               'needed 75). Superseded by d2.',
    ),
    'd2': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='3',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.988),          # 0.997 per frame ~= 0.988 per 4 frames:
                                        # the same ~5.5 s horizon d1 had
        result='STOPPED 2026-10-09 at 6.36M steps (~25M frames): start point '
               '97% -> 78%, 0 fresh-start clears (best 75/102). d1 on env v3 '
               '(4-frame choices + the 10 s stall rule). It never learned to '
               'go straight on the ground (6-9% where only straight was right) '
               'and jumped instead, so it flew past the last spheres '
               '(probe_steering.py). Superseded by d3.',
    ),    'd3': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='4',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.95),           # per CORNER now: ~20 corners, the same
                                        # ~20-cell horizon d2 had at base speed
        result='PARKED 2026-10-09 at 821,424 steps (17.31M frames) for d5: start point '
               '56% since 11.62M frames, fresh-start mean 84.6/102, best 99, 0 clears. Resumed twice '
               'onto env fixes (lag frames; turn window + fresh press). d2 on '
               'env v4: one choice per corner (straight / '
               'left / right / jump), so going straight is one choice like a '
               'turn. d2 never learned ground straights and hopped instead '
               '(probe_steering.py). Same demo, obs, network, stall rule. A '
               'step is now ~16-32 frames, not 4: compare runs in frames.',
    ),    'd4': dict(
        stages=[1, 2, 3, 4], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='5',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817', 'stage2_20261009-170251',
               'stage3_20261009-170931', 'stage4_20261009-172620'],
        demo_prob=0.75,
        ppo=dict(gamma=0.95),
        result='NOT STARTED — proposal, launch only after d3\'s gate and '
               'Tyler\'s go. d3\'s setup on four boards, one demo each, on env '
               'v5 (adds turn-then-jump, which stage 4 needs). Stages 5-7 stay '
               'held out for the zero-shot test (eval_transfer.py).',
    ),
    'd5': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='4',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.99),           # ~100 corners: d3's wall was a jump at
                                        # (26,21) whose penalty lands a median 101
                                        # corners later (0.95^101 = 0.006; 0.99^101
                                        # = 0.36) — probe_steering_v4.py
        result='STOPPED 2026-10-10 05:04 by its stall stop at 2,113,600 steps '
               '(~46.7M frames): passed d3\'s 56% wall at 19.9M frames and its '
               '40M gate at 21%, then 21% unmoved 15.4M frames; 0 fresh clears '
               '(best 100/102). d3 with gamma 0.99 per corner instead of 0.95; '
               'everything else identical, fresh start. d3 stalled at 56%: it '
               'skips blue (25,21) with a P 0.998 jump and seals itself in the '
               'red staircase ~100 corners later, too far for gamma 0.95 to see.',
    ),
    'd7': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='4',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                 SS_POINTER='1'),           # the only change from d5
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.99),
        result='PAUSED by Tyler 2026-10-10 08:01 at 49,200 steps (91%); resume with --resume. d5 plus a compass to the nearest '
               'blue (straight line, heading frame, + distance). d5 stalled at '
               '21% with every leftover blue outside its 15x15 view '
               '(stall_view.py), 71 of 88 still reachable red-free '
               '(stall_reach.py).',
    ),
    'd8': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='4',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                 SS_POINTER='1',
                 SS_SPEED_BONUS='7'),       # the only change from d7
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.99),
        init='d7',                          # first launch starts from d7's save
        result='d7 + a clear-time bonus: a clear also pays 7 per frame under '
               'SS_MAX_STEPS. Tyler 2026-10-10: optimize clear time; his play is '
               'a starting guide, not the target. Starts from d7 at 975,008 '
               'steps (curriculum at ckpt 0; last 300 fresh: 67.7% clears, '
               'median 3,202 frames, best 3,014).',
    ),
    'd8c': dict(
        stages=[1], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='4',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                 SS_POINTER='1'),           # d7 exactly: no clear-time bonus
        demos=['stage1_20260930-165817'], demo_prob=0.75,
        ppo=dict(gamma=0.99),
        init='d7',
        result='CONTROL for d8: d7 with no change, started from d7\'s save '
               'exactly as d8 is, frames from 0. d7 was still getting faster '
               'when paused (median fresh clear 3,310 -> 3,202), so d8 is read '
               'against this at equal frames, not against d7\'s pause.',
    ),
    'd9': dict(
        stages=ALL_14, lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='7',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                 SS_POINTER='1'),           # d7's obs (+ v7's turn_here bit)
        demos=['stage1_20260930-165817', 'stage2_20261009-170251',
               'stage3_20261009-170931', 'stage4_20261009-172620',
               'stage5_20261010-091532', 'stage6_20261010-092841',
               'stage7_20261010-095111', 'stage8_20261010-100029',
               'stage9_20261010-102154', 'stage10_20261010-102523',
               'stage11_20261010-103921', 'stage12_20261010-104326',
               'stage13_20261010-104726', 'stage14_20261010-105034'],
        demo_prob=0.75,
        ppo=dict(gamma=0.99),
        result='STOPPED 2026-10-11 ~16:33 at 5,059,832 steps (~96.5M frames; gate passed, '
               '10/14 boards fresh-cleared) to make way for d10 (Tyler: "Let\'s go ahead and '
               'start it"). LAUNCHED 2026-10-10 ~23:20 by Claude (Tyler gave Claude the box overnight; '
               'criteria AGREED, doc "Run d9"; smoke_corners d9 passed). All 14 boards on env v7, one '
               'recording each (all flawless), d7\'s recipe (compass, gamma '
               '0.99 per decision, backward curriculum). Fresh weights: v7\'s '
               'obs (14 feats) and 9 actions don\'t fit d7\'s network.',
    ),
    'd10': dict(
        stages=ALL_14, lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='7',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                 SS_POINTER='1',
                 SS_SPEED_BONUS='7',        # d8's speed term: a clear pays 7 x (18,000 - frames)
                 # from its next resume (2026-10-16; Tyler: "You can turn those both on"): board 14, the last
                 # without a fresh clear, drawn 6x as often as each other board (~32% of episodes, was ~7%;
                 # 2.7% of the frames); check_stage_weights.py
                 SS_STAGE_WEIGHTS='14:6'),
        demos='book',                       # the route book's current route per board, kept current
        demo_prob=0.75,
        demo_mix=0.3,                       # SS_DEMO_MIX, from the 2026-10-11 ~20:30 resume
        # from its resume at ~7.45M steps (2026-10-16 ~08:30; Tyler: "Pin route, keep d10 going"): the harvest
        # loop's faster board-14 route (2,889 frames, adopted at ~5.08M steps) put the start point at frame 1,351
        # of a route the agent never practised (it goes up a cell and turns back there): no move in 2.4M steps,
        # 0 of 40 probe tries cleared from there. On the old route (2,921) it had been moving back steadily.
        # The finish line is a fresh clear, not the fastest route; the book keeps 2,889 for the main game.
        book_pin={14: 'board14_2921_96f1dc25'},
        ppo=dict(gamma=0.99),
        init='d9',                          # first launch starts from d9's save
        result='DEFINED 2026-10-11 ~15:45 (Tyler: "Yes that sounds great" to starting '
               'from the route book\'s routes; the speed term was the earlier d10 '
               'proposal). d9 + SS_SPEED_BONUS=7, its curriculum along the route '
               'book\'s current best route per board instead of Tyler\'s recordings, '
               'switched live as the book improves. Goal: the book beats Tyler on '
               'all 14 boards (route_book.py). LAUNCHED 2026-10-11 ~16:35 by Claude '
               '(Tyler: "Let\'s go ahead and start it") from d9 at 5,059,832 steps. '
               'demo_mix=0.3 added 2026-10-11 ~20:26 (Tyler: "Yes"): stopped at '
               '1,103,800 steps and resumed, because the late stretches of boards '
               '13 and 14 had faded to ~0 clears beyond the curriculum band.',
    ),
    'd6': dict(
        stages=[1, 2, 3, 4], lstm=512, lr=3e-4,
        obs=dict(SS_ENV_VERSION='5',
                 SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1'),
        demos=['stage1_20260930-165817', 'stage2_20261009-170251',
               'stage3_20261009-170931', 'stage4_20261009-172620'],
        demo_prob=0.75,
        ppo=dict(gamma=0.99),           # d5's gamma; everything else is d4
        result='STOPPED by Tyler 2026-10-10 07:35 at 614,400 steps (~16.5M frames; '
               'start points s1 74% s2 64% s3 79% s4 73%, 0 fresh clears) to make '
               'room for the nearest-sphere pointer. Was the overnight fallback '
               'agreed with Tyler 2026-10-09, launched because d5 got below 56% '
               'and then hit its stall stop. d4 with gamma 0.99 (d5\'s), so d6 vs d5 '
               'isolates training four boards at once.',
    ),
}

# ── Shared config (identical across every run so far — keep it that way) ──────
NUM_ENVS       = 8
AUTOSAVE_STEPS = 100_000       # worst-case loss on a hard kill
CKPT_EVERY     = 800_000       # rolling checkpoints, in env steps (pruned to
                               # last KEEP_CKPTS). Was written as 100_000, but
                               # CheckpointCallback counts calls (NUM_ENVS steps
                               # each), so it was always 800K; now said so.
KEEP_CKPTS     = 5
TOTAL_STEPS    = 150_000_000   # nominal; autosaves + resumes cleanly
HOME           = os.path.dirname(os.path.abspath(__file__))   # the repo, wherever
                                                              # it lives (it moved)

PPO_KWARGS = dict(
    n_steps=1024, batch_size=256, n_epochs=4,
    ent_coef=0.01, clip_range=0.1, gamma=0.997, gae_lambda=0.95,
)
LSTM_KWARGS = dict(n_lstm_layers=2, shared_lstm=False, enable_critic_lstm=True)

# Demo curriculum (runs with `demos`): when to move a demo's start point back.
DEMO_WINDOW    = 32     # recent episodes judged at the current start point
DEMO_MIN_N     = 16     # ...needed before the start point may move
DEMO_ADVANCE   = 0.25   # move back once >= 25% of them matched or beat the demo
BOOK_CHECK_S   = 300    # demos='book': look for a faster route this often (seconds)


def paths(run):
    # absolute, so launching from another directory can't miss an existing
    # checkpoint and silently start "fresh"
    return dict(
        zip=os.path.join(HOME, f'sonic_ss_multistage_{run}'),
        pkl=os.path.join(HOME, f'sonic_ss_multistage_{run}_vecnormalize.pkl'),
        ckpt_dir=os.path.join(HOME, f'checkpoints_ss_multistage_{run}'),
        stop=os.path.join(HOME, f'STOP_{run}'),
        log=os.path.join(HOME, f'ss_log_{run}.jsonl'),
        demo_state=os.path.join(HOME, f'sonic_ss_multistage_{run}_demos.json'),
        prefix=f'sonic_ss_ms_{run}',
    )


def book_demos(stages, pins=None):
    """demos='book': the route book's current route for each board
    (route_book.py), copied to routes/curriculum/board{NN}_{frames}_{hash}.npz
    so that a demo's id names one exact route: the curriculum's start points
    belong to a route, and the book's file for a board changes as it improves.
    pins: {board: curriculum demo id} (RUNS 'book_pin'): train that board on
    that route whatever the book holds (the book itself is left alone)."""
    import hashlib
    import route_book
    out = []
    for b in stages:
        if pins and b in pins:
            pinned = os.path.join(HOME, 'routes', 'curriculum', f'{pins[b]}.npz')
            if not os.path.exists(pinned):
                raise FileNotFoundError(f'book_pin for board {b}: no {pinned}')
            out.append(pinned)
            continue
        src = route_book.book_path(b)
        with open(src, 'rb') as f:
            h = hashlib.sha1(f.read()).hexdigest()[:8]
        import demos as demo_io
        n = len(demo_io.load_demo(src)[0])
        dst = os.path.join(HOME, 'routes', 'curriculum', f'board{b:02d}_{n}_{h}.npz')
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst + '.tmp.npz')
            os.replace(dst + '.tmp.npz', dst)
        out.append(dst)
    return out


def demo_paths(cfg):
    """A run's demo files: its `demos` patterns, matched under demos/, or
    'book' (book_demos)."""
    if cfg.get('demos') == 'book':
        return book_demos(cfg['stages'], cfg.get('book_pin'))
    found = []
    for pat in cfg.get('demos', []):
        found += sorted(glob.glob(os.path.join(HOME, 'demos', f'{pat}.npz')))
    return found


def list_runs():
    print(f'{"run":6} {"stages":8} {"lstm":6} {"lr":8} {"obs":41} status')
    for name, cfg in RUNS.items():
        p = paths(name)
        obs = cfg['obs']
        obs_s = (f"env=v{obs['SS_ENV_VERSION']} pix={obs['SS_USE_PIXELS']} "
                 f"patch_r={obs['SS_PATCH_R']} droppos={obs['SS_DROP_POSITION']}")
        if os.path.exists(f"{p['zip']}.zip"):
            status = f"checkpoint on disk ({os.path.getsize(p['zip'] + '.zip') / 1e6:.0f} MB)"
        else:
            status = 'no checkpoint'
        print(f'{name:6} {len(cfg["stages"]):<8} {cfg["lstm"]:<6} '
              f'{cfg["lr"]:<8.0e} {obs_s:41} {status}')
        if cfg.get('demos'):
            print(f'       demos {cfg["demos"]} at {cfg.get("demo_prob", 0.5):.0%}: '
                  f'{len(demo_paths(cfg))} recorded')
        print(f'       {cfg["result"]}')


# ── Argument parsing MUST happen before importing the env: the env reads its
#    observation shape from SS_* at module import, and SubprocVecEnv workers
#    inherit this process's environment. ────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('run', nargs='?', help='run name from RUNS (e.g. v5)')
ap.add_argument('--list', action='store_true', help='show runs and exit')
ap.add_argument('--resume', action='store_true',
                help='continue an existing checkpoint. Required whenever one is '
                     'on disk, so a "fresh" launch can never silently resume.')
ap.add_argument('--restart-prob', type=float, default=0.0,
                help='Go-Explore restart probability. 0 for every multistage '
                     'run so far. Snapshots now record their stage, so '
                     'multistage restarts are possible — but untested.')
args = ap.parse_args()

if args.list or not args.run:
    list_runs()
    sys.exit(0 if args.list else 2)

if args.run not in RUNS:
    sys.exit(f'unknown run {args.run!r}. Known: {", ".join(RUNS)} '
             f'(add a new entry to RUNS rather than copying this file).')

CFG = RUNS[args.run]
P = paths(args.run)

os.environ.update(CFG['obs'])
os.environ['SS_STAGES'] = ','.join(str(s) for s in CFG['stages'])
os.environ['SS_RESTART_PROB'] = str(args.restart_prob)
os.environ['SS_LOG'] = P['log']          # per-run log: a shared log made every
                                         # analyzer read the GHOST of prior runs
DEMOS = demo_paths(CFG)
if CFG.get('demos') and not DEMOS:
    sys.exit(f"\nrun {args.run} learns from demos {CFG['demos']}, but none are "
             f"recorded yet in {os.path.join(HOME, 'demos')}.\n"
             f"  record one:  python3 record_demo.py <stage>\n")
os.environ['SS_DEMOS'] = ','.join(DEMOS)
os.environ['SS_DEMO_PROB'] = str(CFG.get('demo_prob', 0.5))
os.environ['SS_DEMO_MIX'] = str(CFG.get('demo_mix', 0))   # not in obs: the pilot never starts on a demo

import torch                                                    # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.callbacks import (BaseCallback,   # noqa: E402
                                                CheckpointCallback,
                                                CallbackList)
try:
    from sb3_contrib import RecurrentPPO                        # noqa: E402
except ImportError:
    raise ImportError('pip install sb3-contrib --break-system-packages')

import special_stage_env                                        # noqa: E402
from special_stage_env import SpecialStageEnv                   # noqa: E402

# set by the signal handler; polled by the GracefulStop callback
_STOP = {'requested': False, 'why': ''}


def _request_stop(signum, frame):
    _STOP['requested'] = True
    _STOP['why'] = f'signal {signum}'
    print(f'\n[GracefulStop] caught signal {signum} — will save and exit '
          f'at the next step boundary.', flush=True)


signal.signal(signal.SIGTERM, _request_stop)   # kill / pkill (parent)
signal.signal(signal.SIGINT, _request_stop)    # Ctrl-C


def save_all(model, venv, tag, steps):
    """Save policy + VecNormalize together. Safe to call repeatedly."""
    try:
        model.save(P['zip'])
        venv.save(P['pkl'])
        print(f"[{tag}] saved at {steps} steps -> {P['zip']}.zip + {P['pkl']}",
              flush=True)
        return True
    except Exception as e:
        print(f'[{tag}] SAVE FAILED at {steps} steps: {e!r}', flush=True)
        return False


class AutoSaveCallback(BaseCallback):
    """Periodic save of the main checkpoint used for resume."""
    def __init__(self, save_freq=AUTOSAVE_STEPS, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            save_all(self.model, self.training_env, 'AutoSave', self.num_timesteps)
        return True


class GracefulStopCallback(BaseCallback):
    """
    Stop cleanly on either a signal (SIGTERM/SIGINT) or the STOP sentinel file.
    Saves, then returns False so model.learn() unwinds normally.
    Polls the file every `poll_every` steps to avoid a stat() on every step.
    """
    def __init__(self, stop_file, poll_every=200, verbose=0):
        super().__init__(verbose)
        self.stop_file = stop_file
        self.poll_every = poll_every
        self._last_poll = 0

    def _on_step(self):
        if not _STOP['requested']:
            if self.num_timesteps - self._last_poll >= self.poll_every:
                self._last_poll = self.num_timesteps
                if os.path.exists(self.stop_file):
                    _STOP['requested'] = True
                    _STOP['why'] = f'stop file {self.stop_file}'
        if _STOP['requested']:
            print(f"\n[GracefulStop] stopping ({_STOP['why']}) — saving now...",
                  flush=True)
            save_all(self.model, self.training_env, 'GracefulStop',
                     self.num_timesteps)
            # remove the sentinel so the next launch isn't immediately stopped
            try:
                if os.path.exists(self.stop_file):
                    os.remove(self.stop_file)
                    print(f'[GracefulStop] removed {self.stop_file}', flush=True)
            except OSError as e:
                print(f'[GracefulStop] could not remove stop file: {e!r}',
                      flush=True)
            return False        # <- tells learn() to stop
        return True


class PruneCheckpointsCallback(BaseCallback):
    def __init__(self, checkpoint_dir, keep=KEEP_CKPTS, every=CKPT_EVERY, verbose=0):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.keep = keep
        self.every = every
        self.last = 0

    def _on_step(self):
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        files = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.zip')),
                       key=os.path.getmtime)
        for f in files[:-self.keep]:
            try:
                os.remove(f)
            except OSError:
                pass
        return True


class DemoCurriculumCallback(BaseCallback):
    """Backward curriculum over the recorded demos (Salimans & Chen 2018).

    A share of episodes start from a state along a demo, at or just after that
    demo's start point. The start point begins near the END of the demo and
    moves one checkpoint (~2 cells) toward the stage start whenever enough
    recent episodes from it matched or beat the demo — collected at least as
    many spheres from there as the demo did. At checkpoint 0 the agent is
    playing the stage from the start, measured against the demo's whole run.
    Only the demo's states are used, never its inputs, so the agent can end up
    better than the demo. The start points persist next to the checkpoint, so
    --resume keeps the curriculum where it was.

    A demo new to the curriculum (demos='book': a faster route; or the d9
    recordings d10 starts from) takes over its board's start point at the same
    FRAME (the first start point at or after it): a route spliced onto an
    older one shares its frames up to the splice, so there it is the very
    state the agent had mastered; past the splice it is the agent's own run.
    With demos='book' the book is checked every BOOK_CHECK_S seconds; when a
    board's route changed, every worker switches at its next reset (env
    set_demo_files, so no episode is disturbed), and once worker 0 has the new
    set its start points are carried over and pushed."""

    def __init__(self, state_path, book_stages=None, verbose=0):
        super().__init__(verbose)
        self.state_path = state_path
        self.book_stages = book_stages
        self.ptr, self.fracs, self.recent = {}, {}, {}
        self.frames, self.stage = {}, {}
        self._book_t = 0.0
        self._pending = None            # demo ids the workers are switching to

    @staticmethod
    def _board_of(did, entry):
        if 'stage' in entry:
            return int(entry['stage'])
        import re
        m = re.match(r'(?:stage|board)0*(\d+)_', did)
        return int(m.group(1)) if m else None

    @staticmethod
    def _frame_of(did, entry):
        """The start point's frame in its demo ('frame', or frac x the length of
        demos/<id>.npz for state files written before 'frame' was kept)."""
        if 'frame' in entry:
            return entry['frame']
        p = os.path.join(HOME, 'demos', f'{did}.npz')
        if os.path.exists(p):
            import demos as demo_io
            return entry['frac'] * len(demo_io.load_demo(p)[0])
        return None

    def _adopt(self, infos, saved):
        """Start points for these demos: their own if saved, else their board's
        old one carried over by frame, else the end."""
        by_board = {}
        for sid, e in saved.items():
            b, f = self._board_of(sid, e), self._frame_of(sid, e)
            if b is not None and f is not None:
                by_board[b] = (sid, f)
        for d in infos:
            did, last = d['id'], len(d['fracs']) - 1
            self.fracs[did], self.frames[did], self.stage[did] = d['fracs'], d['frames'], d['stage']
            note = ''
            if did in saved:
                p = min(saved[did].get('ptr', last), last)
            elif d['stage'] in by_board:
                sid, f = by_board[d['stage']]
                steps = [fr * d['frames'] for fr in d['fracs']]
                p = next((i for i, s in enumerate(steps) if s >= f - 0.5), last)
                note = f' (carried over from {sid} at frame {f:.0f})'
            else:
                p = last
            self.ptr[did] = p
            self.recent[did] = collections.deque(maxlen=DEMO_WINDOW)
            print(f"[demo] {did}: stage {d['stage']}, {d['final']} spheres in "
                  f"{d['frames']} frames; episodes start "
                  f"{d['fracs'][p]:.0%} of the way in{note}", flush=True)

    def _on_training_start(self):
        saved = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                saved = json.load(f)
        self._adopt(self.training_env.env_method('demo_info', indices=[0])[0], saved)
        self._book_t = time.time()
        self._push()

    def _state(self):
        return {k: {'ptr': p, 'frac': round(self.fracs[k][p], 4), 'stage': self.stage[k],
                    'frame': round(self.fracs[k][p] * self.frames[k])}
                for k, p in self.ptr.items()}

    def _push(self):
        self.training_env.env_method('set_demo_pointers', dict(self.ptr))
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._state(), f, indent=1)
        os.replace(tmp, self.state_path)

    def _check_book(self):
        """demos='book': a board's route changed -> the workers switch to the
        book's current set at their next reset."""
        self._book_t = time.time()
        try:
            new = book_demos(self.book_stages, CFG.get('book_pin'))
        except Exception as e:
            print(f'[demo] route book unreadable ({e!r}); keeping the current routes', flush=True)
            return
        ids = {os.path.splitext(os.path.basename(p))[0] for p in new}
        if ids == (self._pending or set(self.ptr)):
            return
        self.training_env.env_method('set_demo_files', new)
        self._pending = ids
        print(f"[demo] the route book has a faster route: "
              f"{', '.join(sorted(ids - set(self.ptr)))} (switching at each worker's next episode)",
              flush=True)

    def _adopt_pending(self):
        """Once worker 0 has rebuilt its demos: drop the replaced routes and
        carry their start points over to the new ones."""
        infos = self.training_env.env_method('demo_info', build=False, indices=[0])[0]
        if not infos or {d['id'] for d in infos} != self._pending:
            return
        old = self._state()
        gone = [k for k in self.ptr if k not in self._pending]
        for k in gone:
            for dct in (self.ptr, self.fracs, self.recent, self.frames, self.stage):
                dct.pop(k, None)
        self._adopt([d for d in infos if d['id'] not in self.ptr], {k: old[k] for k in gone})
        self._pending = None
        self._push()

    def _on_step(self):
        moved = False
        for info in self.locals.get('infos', []):
            r = info.get('demo_result')
            if not r or r['demo'] not in self.ptr:
                continue
            p = self.ptr[r['demo']]
            if not p <= r['start'] <= p + special_stage_env.DEMO_BAND:
                continue                    # began before the start point moved
            q = self.recent[r['demo']]
            q.append(r)
            if len(q) >= DEMO_MIN_N and p > 0:
                rate = sum(x['success'] for x in q) / len(q)
                if rate >= DEMO_ADVANCE:
                    margin = sum(x['gained'] - x['target'] for x in q) / len(q)
                    self.ptr[r['demo']] = p - 1
                    q.clear()
                    moved = True
                    fr = self.fracs[r['demo']]
                    print(f"[demo] {r['demo']}: start -> {fr[p - 1]:.0%} into the "
                          f"demo ({rate:.0%} matched or beat it from {fr[p]:.0%}; "
                          f"mean margin {margin:+.1f} spheres)", flush=True)
        if moved:
            self._push()
        return True

    def _on_rollout_end(self):
        for did, p in self.ptr.items():
            self.logger.record(f'demo/{did}/start_frac', self.fracs[did][p])
        if self._pending:
            self._adopt_pending()
        if self.book_stages and time.time() - self._book_t >= BOOK_CHECK_S:
            self._check_book()


def make_env():
    def _init():
        return SpecialStageEnv()
    return _init


if __name__ == '__main__':
    resuming = os.path.exists(f"{P['zip']}.zip")

    print(f"=== multistage run {args.run} ===")
    print(f"  {CFG['result']}")
    print(f"  stages       : {CFG['stages']}")
    print(f"  lstm_hidden  : {CFG['lstm']}   lr: {CFG['lr']:.0e}   "
          f"ppo: {({**PPO_KWARGS, **CFG.get('ppo', {})})}")
    print(f"  obs preset   : env=v{special_stage_env.ENV_VERSION} "
          f"pixels={special_stage_env.USE_PIXELS} "
          f"patch={special_stage_env.PATCH_SIZE}x{special_stage_env.PATCH_SIZE}"
          f"x{special_stage_env.GRID_CHANNELS} "
          f"ram_feats={special_stage_env.NUM_RAM_FEATURES} "
          f"speed_bonus={special_stage_env.SPEED_BONUS:g}")
    print(f"  restart_prob : {args.restart_prob}")
    if DEMOS:
        print(f"  demos        : {len(DEMOS)} at {CFG.get('demo_prob', 0.5):.0%} of "
              f"episodes: {', '.join(os.path.basename(d) for d in DEMOS)}")
        if special_stage_env.DEMO_MIX:
            print(f"  demo mix     : {special_stage_env.DEMO_MIX:.0%} of demo starts anywhere from "
                  f"the start point to the end (else within {special_stage_env.DEMO_BAND} of it)")
    print(f"  log          : {P['log']}")
    print(f"  autosave every {AUTOSAVE_STEPS:,} steps | "
          f"graceful stop: touch {P['stop']}")
    print(f'  CUDA available: {torch.cuda.is_available()}')

    # A stale checkpoint makes a "fresh" run silently RESUME. This bit the v2
    # launch: a 1M-step partial resumed and the per-stage curve looked like
    # catastrophic interference. Refuse rather than warn.
    if resuming and not args.resume:
        sys.exit(
            f"\nREFUSING TO LAUNCH: {P['zip']}.zip already exists, so this would "
            f"RESUME, not start fresh.\n"
            f"  - to continue it:      add --resume\n"
            f"  - to restart it clean: mv {P['zip']}.zip {P['pkl']} aside first\n"
            f"  - for a NEW experiment: add a new entry to RUNS\n")
    if args.resume and not resuming:
        sys.exit(f"\n--resume passed but no {P['zip']}.zip exists. Drop --resume "
                 f"to start {args.run} fresh.\n")

    # Checked here as well as in the env: a worker that dies on a missing file
    # surfaces as an opaque SubprocVecEnv pipe error.
    missing = [f for f in special_stage_env.STAGE_FILES if not os.path.exists(f)]
    if missing:
        sys.exit('\nREFUSING TO LAUNCH: stage state file(s) missing:\n  '
                 + '\n  '.join(missing) + '\n')

    # Each demo must still replay into the run it recorded (same stage state,
    # same action table); the workers would fail on it less legibly.
    if DEMOS:
        import demos as demo_io
        for d in DEMOS:
            ok, note = demo_io.replay_check(d)
            print(f'  demo check   : {os.path.basename(d)}: {note}')
            if not ok:
                sys.exit(f'\nREFUSING TO LAUNCH: {d} no longer replays.\n')

    tb_dir = os.path.join(HOME, 'logs_ss')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(P['ckpt_dir'], exist_ok=True)

    # a stale sentinel from a previous stop would kill this run instantly
    if os.path.exists(P['stop']):
        os.remove(P['stop'])
        print(f"[startup] cleared stale {P['stop']}")

    # `init`: a run's first launch starts from another run's save (weights,
    # VecNormalize, curriculum pointer) instead of random weights; its frame
    # count and log start at zero. --resume afterwards uses its own save.
    init = CFG.get('init') if not resuming else None
    if init:
        src = paths(init)
        for f in (f"{src['zip']}.zip", src['pkl']):
            if not os.path.exists(f):
                sys.exit(f'\nREFUSING TO LAUNCH: {args.run} starts from {init}, '
                         f'but {f} is missing.\n')
        if os.path.exists(src['demo_state']) and DEMOS:
            shutil.copyfile(src['demo_state'], P['demo_state'])
        print(f"  init         : starting from {init}'s save "
              f"({src['zip']}.zip, its VecNormalize and curriculum pointer)")

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    if init:
        env = VecNormalize.load(paths(init)['pkl'], env)
        env.training = True
        print(f"Loaded VecNormalize from {paths(init)['pkl']}")
    elif os.path.exists(P['pkl']):
        env = VecNormalize.load(P['pkl'], env)
        env.training = True
        print(f"Resumed VecNormalize from {P['pkl']}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0)
        print(f'Created fresh VecNormalize (multistage {args.run})')

    if resuming:
        _lr = os.environ.get('LR_OVERRIDE')
        _co = {'learning_rate': float(_lr)} if _lr else None
        model = RecurrentPPO.load(P['zip'], env=env, device='auto',
                                  custom_objects=_co)
        if _lr:
            # force it onto the live objects too (some SB3 versions need this)
            model.learning_rate = float(_lr)
            model.lr_schedule = lambda _: float(_lr)
        print(f'Resuming multistage {args.run} at {model.num_timesteps} steps '
              f'| gamma {model.gamma} lr {model.learning_rate}'
              + (f' | LR OVERRIDDEN -> {_lr}' if _lr else ''))
    elif init:
        model = RecurrentPPO.load(f"{paths(init)['zip']}.zip", env=env, device='auto',
                                  tensorboard_log=tb_dir)
        for k, v in {'learning_rate': CFG['lr'], **PPO_KWARGS, **CFG.get('ppo', {})}.items():
            got = getattr(model, k)
            got = got(1.0) if callable(got) else got      # clip_range loads as a schedule
            if got != v:
                sys.exit(f'\nREFUSING TO LAUNCH: {init} has {k}={got}, {args.run} wants {v}.\n')
        print(f'Starting {args.run} from {init} (its weights at {model.num_timesteps} '
              f'steps; {args.run} counts from 0) | gamma {model.gamma} lr {model.learning_rate}')
    else:
        model = RecurrentPPO(
            policy='MultiInputLstmPolicy',
            env=env,
            learning_rate=CFG['lr'],
            policy_kwargs=dict(lstm_hidden_size=CFG['lstm'], **LSTM_KWARGS),
            verbose=1,
            tensorboard_log=tb_dir,
            device='auto',
            **{**PPO_KWARGS, **CFG.get('ppo', {})},   # a run may override
        )
        print(f'Starting FRESH multistage {args.run} model (random weights)')

    callbacks = CallbackList([
        GracefulStopCallback(P['stop']),              # checked first
        # save_freq counts callback calls, and each call is NUM_ENVS env steps
        CheckpointCallback(save_freq=max(CKPT_EVERY // NUM_ENVS, 1),
                           save_path=P['ckpt_dir'],
                           name_prefix=P['prefix'], verbose=1),
        AutoSaveCallback(save_freq=AUTOSAVE_STEPS),
        PruneCheckpointsCallback(checkpoint_dir=P['ckpt_dir']),
    ] + ([DemoCurriculumCallback(P['demo_state'], book_stages=(
        CFG['stages'] if CFG.get('demos') == 'book' else None))] if DEMOS else []))

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks,
                    reset_num_timesteps=bool(init))   # an init run counts from 0
    except BaseException as e:
        # crash, KeyboardInterrupt that slipped through, anything — save first
        print(f'\n[exit] {type(e).__name__}: {e!r} — emergency save...', flush=True)
        save_all(model, env, 'EmergencySave', model.num_timesteps)
        raise
    finally:
        # normal completion or graceful stop: one last save so disk == memory
        save_all(model, env, 'FinalSave', model.num_timesteps)
        try:
            env.close()
        except Exception:
            pass
