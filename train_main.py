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
import signal
import argparse
import collections

import numpy as np
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                CallbackList)

# ── Run registry ─────────────────────────────────────────────────────────────
# `acts` are act names as coordinator.act_name makes them ('AIZ1', 'HCZ2'...);
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

    def __init__(self, state_path, band):
        super().__init__()
        self.state_path, self.band = state_path, band
        self.ptr, self.fracs, self.recent = {}, {}, {}

    def _on_training_start(self):
        saved = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                saved = json.load(f)
        for d in self.training_env.env_method('demo_info', indices=[0])[0]:
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
            if not r or r['key'] not in self.ptr:
                continue
            p = self.ptr[r['key']]
            if not p <= r['start'] <= p + self.band:
                continue                    # began before the pointer moved
            q = self.recent[r['key']]
            q.append(r)
            if len(q) >= CURR_MIN_N and p > 0:
                rate = sum(x['success'] for x in q) / len(q)
                if rate >= CURR_ADVANCE:
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
    import demos as demo_io

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

    # every recording must still replay into the run it recorded
    for d in DEMOS:
        ok, note = demo_io.replay_play_check(d)
        print(f'  demo check   : {os.path.basename(d)}: {note}', flush=True)
        if not ok:
            sys.exit(f'\nREFUSING TO LAUNCH: {d} no longer replays.\n')

    os.makedirs(P['ckpt_dir'], exist_ok=True)
    tb_dir = os.path.join(HOME, 'logs_main')
    os.makedirs(tb_dir, exist_ok=True)
    if os.path.exists(P['stop']):
        os.remove(P['stop'])

    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    if os.path.exists(P['pkl']):
        env = VecNormalize.load(P['pkl'], env)
        env.training = True
        print(f"Resumed VecNormalize from {P['pkl']}")
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
        ActCurriculumCallback(P['curriculum'], coordinator.DEMO_BAND),
    ])
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
