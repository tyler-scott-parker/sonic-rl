"""
train_ss.py — the SINGLE-STAGE specialist trainer (sonic_ss_model_v2).

PAUSED at ~326M steps and preserved as the baseline/control: the run that proved
one board is winnable (fresh best 96/102, 3 wins from restarts) and, by
memorizing that board (transfer ratio 0.06), made the generalization findings
legible. Multistage runs live in train_ss_multistage.py.

  - learning_rate cooled 3e-4 -> 1e-4 -> 5e-5 on resume via custom_objects
    (the optimizer was running hot: approx_kl 0.05-0.09, clip_fraction ~0.5)
  - gamma 0.99 -> 0.997 the same way; a constructor edit is a NO-OP on resume

Runs the pre-v4 observation (pixels + 11x11 patch + absolute position), which is
what this checkpoint was trained on. The preset is pinned below — do NOT rely on
the env defaults, which are the multistage ones.

LAUNCH:
  cd ~/sonic-rl && source venv/bin/activate
  nohup python3 train_ss.py > logs/ss_trainer.log 2>&1 &
  grep -m1 "Resuming" logs/ss_trainer.log     # shows the LIVE lr AND gamma
"""

import os

# The specialist's observation preset. Must be set before special_stage_env is
# imported (it reads its obs shape at import) and before SubprocVecEnv forks.
os.environ.setdefault('SS_USE_PIXELS', '1')
os.environ.setdefault('SS_PATCH_R', '5')
os.environ.setdefault('SS_DROP_POSITION', '0')

import glob
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError(
        "sb3-contrib not installed. Run:\n"
        "pip install sb3-contrib --break-system-packages"
    )

from special_stage_env import SpecialStageEnv

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ENVS        = 8
PKL             = 'sonic_ss_vecnormalize_v2.pkl'
ZIP             = 'sonic_ss_model_v2'
SAVE_FREQ       = 500_000
TOTAL_STEPS     = 200_000_000
RESUME_LR       = 5e-5          # cooled from 3e-4 for the mature phase
RESUME_GAMMA    = 0.997         # was 0.99 in the pickle; ~1-sphere horizon was
                                # too myopic for endgame route-planning. Must go
                                # through custom_objects — the constructor never
                                # runs on a resume, so editing gamma there is a
                                # no-op (this is why the earlier change never took).

# ── Callbacks ─────────────────────────────────────────────────────────────────
class AutoSaveCallback(BaseCallback):
    def __init__(self, save_freq=SAVE_FREQ, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.model.save(ZIP)
            self.training_env.save(PKL)
            self.last_save = self.num_timesteps
            print(f'[AutoSave] Saved at {self.num_timesteps} steps')
        return True


class PruneCheckpointsCallback(BaseCallback):
    """Keep only the N most recent checkpoints; runs on autosave cadence."""
    def __init__(self, checkpoint_dir, keep=5, every=100_000, verbose=0):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.keep = keep
        self.every = every
        self.last = 0

    def _on_step(self):
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        files = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, '*.zip')),
            key=os.path.getmtime
        )
        for f in files[:-self.keep]:
            try:
                os.remove(f)
            except OSError:
                pass
        return True


# ── Environment factory ───────────────────────────────────────────────────────
def make_env():
    def _init():
        return SpecialStageEnv()
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')

    os.makedirs('./logs_ss/', exist_ok=True)
    os.makedirs('./checkpoints_ss/', exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    if os.path.exists(PKL):
        env = VecNormalize.load(PKL, env)
        env.training = True
        print(f'Loaded VecNormalize from {PKL}')
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0)
        print('Created fresh VecNormalize')

    if os.path.exists(f'{ZIP}.zip'):
        model = RecurrentPPO.load(
            ZIP, env=env, device='auto',
            custom_objects={'learning_rate': RESUME_LR, 'gamma': RESUME_GAMMA},
        )
        print(f'Resuming from {ZIP} at {model.num_timesteps} steps, '
              f'lr={model.learning_rate}')
    else:
        model = RecurrentPPO(
            policy='MultiInputLstmPolicy',
            env=env,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.1,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=dict(
                lstm_hidden_size=512,
                n_lstm_layers=2,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            verbose=1,
            tensorboard_log='./logs_ss/',
            device='auto'
        )
        print('Starting fresh SS model')

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./checkpoints_ss/',
        name_prefix='sonic_ss',
        verbose=1,
    )

    autosave_callback = AutoSaveCallback(save_freq=SAVE_FREQ)
    prune_callback = PruneCheckpointsCallback(checkpoint_dir='./checkpoints_ss/', keep=5)

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([checkpoint_callback, autosave_callback, prune_callback])

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=False
    )
