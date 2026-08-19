"""
train_ss_multistage_v3.py — the 14-board run. (graceful-stop edition)

TRAIN: all 14 special stages. No holdout — the game has exactly 14 fixed stages,
so an agent that memorizes all 14 is behaviorally identical to a generalist for
this use case. (The generalization question stays open; see sonic_rl_context.md.)

Position stays DROPPED (SS_DROP_POSITION=1), network stays 512-hidden/2-layer.

LAUNCH:
  cd ~/sonic-rl && source venv/bin/activate
  ls sonic_ss_multistage_v5* 2>/dev/null && echo "MOVE ASIDE FIRST"
  SS_LOG=~/sonic-rl/ss_log_v3.jsonl \
  SS_DROP_POSITION=1 SS_STAGES=1,2,3,4,5,6,7,8,9,10,11,12,13,14 SS_RESTART_PROB=0 \
    nohup python3 train_ss_multistage_v3.py > logs/ss_multistage_v5.log 2>&1 &
  grep -E "FRESH|Resuming|SS_STAGES" logs/ss_multistage_v5.log

STOPPING — three ways, best first:
  1. GRACEFUL (recommended, loses NOTHING):
         touch ~/sonic-rl/STOP_v5
     The trainer notices within one step, saves model + VecNormalize, exits.
     Watch for "[GracefulStop] saved at N steps" in the log, then the file is
     auto-removed so your next launch isn't immediately stopped.
  2. Ctrl-C / kill <parent PID>  -> signal handler saves, then exits.
  3. pkill -f train_ss_multistage_v3.py -> WORKS but is the least safe: pkill
     also matches the SubprocVecEnv workers (they inherit the parent's cmdline),
     so workers may die before the parent finishes saving. Prefer method 1.

Autosaves every AUTOSAVE_STEPS (100K) regardless, so even a hard kill -9 costs
at most ~100K steps.
"""

import os
import glob
import signal
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                CallbackList)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("pip install sb3-contrib --break-system-packages")

from special_stage_env_v4 import SpecialStageEnv   # v4: pixel-free, wider patch, 11 RAM feats

# ── Config ───────────────────────────────────────────────────────────────────
NUM_ENVS       = 8
PKL            = 'sonic_ss_multistage_v5_vecnormalize.pkl'
ZIP            = 'sonic_ss_multistage_v5'
AUTOSAVE_STEPS = 100_000       # was 500_000 — worst-case loss on a hard kill
CKPT_EVERY     = 100_000       # rolling checkpoints (pruned to last 5)
TOTAL_STEPS    = 150_000_000   # nominal; autosaves + resumes cleanly
CKPT_DIR       = './checkpoints_ss_multistage_v5/'
STOP_FILE      = os.path.expanduser('~/sonic-rl/STOP_v5')

# set by the signal handler; polled by the GracefulStop callback
_STOP = {'requested': False, 'why': ''}


def _request_stop(signum, frame):
    _STOP['requested'] = True
    _STOP['why'] = f'signal {signum}'
    print(f'\n[GracefulStop] caught signal {signum} — will save and exit '
          f'at the next step boundary.', flush=True)


signal.signal(signal.SIGTERM, _request_stop)   # kill / pkill (parent)
signal.signal(signal.SIGINT, _request_stop)    # Ctrl-C

print(f"SS_LOG={os.environ.get('SS_LOG','(default ss_log.jsonl)')}")
print(f"SS_DROP_POSITION={os.environ.get('SS_DROP_POSITION','0')}  "
      f"SS_STAGES={os.environ.get('SS_STAGES','(unset)')}  "
      f"SS_RESTART_PROB={os.environ.get('SS_RESTART_PROB','0.25')}")
print(f"autosave every {AUTOSAVE_STEPS:,} steps | "
      f"graceful stop: touch {STOP_FILE}")


def save_all(model, venv, tag, steps):
    """Save policy + VecNormalize together. Safe to call repeatedly."""
    try:
        model.save(ZIP)
        venv.save(PKL)
        print(f'[{tag}] saved at {steps} steps -> {ZIP}.zip + {PKL}', flush=True)
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
    Polls the file every `poll_every` steps to avoid a stat() on every single step.
    """
    def __init__(self, stop_file=STOP_FILE, poll_every=200, verbose=0):
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
    def __init__(self, checkpoint_dir, keep=5, every=CKPT_EVERY, verbose=0):
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


def make_env():
    def _init():
        return SpecialStageEnv()
    return _init


if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    os.makedirs('./logs_ss/', exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # a stale sentinel from a previous stop would kill this run instantly
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
        print(f'[startup] cleared stale {STOP_FILE}')

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    if os.path.exists(PKL):
        env = VecNormalize.load(PKL, env)
        env.training = True
        print(f'Resumed VecNormalize from {PKL}')
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0)
        print('Created fresh VecNormalize (multistage v5)')

    if os.path.exists(f'{ZIP}.zip'):
        _lr = os.environ.get('LR_OVERRIDE')
        _co = {'learning_rate': float(_lr)} if _lr else None
        model = RecurrentPPO.load(ZIP, env=env, device='auto', custom_objects=_co)
        if _lr:
            # force it onto the live objects too (some SB3 versions need this)
            model.learning_rate = float(_lr)
            model.lr_schedule = lambda _: float(_lr)
        print(f'Resuming multistage v5 at {model.num_timesteps} steps'
              + (f' | LR OVERRIDDEN -> {_lr}' if _lr else ''))
    else:
        model = RecurrentPPO(
            policy='MultiInputLstmPolicy',
            env=env,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            learning_rate=2e-4,   # v5: mild reduction for the bigger net; watch clip_fraction/KL early
            ent_coef=0.01,
            clip_range=0.1,
            gamma=0.997,
            gae_lambda=0.95,
            policy_kwargs=dict(
                lstm_hidden_size=1024,   # v5: 2x capacity — THE variable being tested vs v4's 512
                n_lstm_layers=2,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            verbose=1,
            tensorboard_log='./logs_ss/',
            device='auto',
        )
        print('Starting FRESH multistage v5 model (random weights)')

    callbacks = CallbackList([
        GracefulStopCallback(),                       # checked first
        CheckpointCallback(save_freq=CKPT_EVERY, save_path=CKPT_DIR,
                           name_prefix='sonic_ss_ms_v5', verbose=1),
        AutoSaveCallback(save_freq=AUTOSAVE_STEPS),
        PruneCheckpointsCallback(checkpoint_dir=CKPT_DIR, keep=5),
    ])

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks,
                    reset_num_timesteps=False)
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
