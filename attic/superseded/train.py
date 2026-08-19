"""
train_ss.py — Dedicated special stage trainer.

Uses RecurrentPPO (LSTM) + MultiInputLstmPolicy + Dict observations:
  - pixels: 84x84 RGB color (blue/red spheres distinguishable)
  - ram:    6 normalized RAM features

Separate from the main game model. Runs independently.
"""

import os
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
NUM_ENVS        = 8          # fewer than main game — LSTM needs consistent episodes
PKL             = 'sonic_ss_vecnormalize.pkl'
ZIP             = 'sonic_ss_model'
SAVE_FREQ       = 500_000    # autosave every 500k steps
TOTAL_STEPS     = 200_000_000

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
    """Keep only the N most recent checkpoints in the checkpoint directory."""
    def __init__(self, checkpoint_dir, keep=5, verbose=0):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.keep = keep

    def _on_step(self):
        files = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, '*.zip')),
            key=os.path.getmtime
        )
        for f in files[:-self.keep]:
            os.remove(f)
        return True


# ── Environment factory ───────────────────────────────────────────────────────
def make_env():
    def _init():
        return SpecialStageEnv()
    return _init


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')

    os.makedirs('./logs_ss/', exist_ok=True)
    os.makedirs('./checkpoints_ss/', exist_ok=True)

    # Build vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # VecNormalize — normalize rewards only, not observations
    # (pixels are normalized by CNN, RAM features are pre-normalized)
    if os.path.exists(PKL):
        env = VecNormalize.load(PKL, env)
        env.training = True
        print(f'Loaded VecNormalize from {PKL}')
    else:
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0
        )
        print('Created fresh VecNormalize')

    # Build or resume model
    if os.path.exists(f'{ZIP}.zip'):
        model = RecurrentPPO.load(
            ZIP, env=env, device='auto',
        )
        print(f'Resuming from {ZIP} at {model.num_timesteps} steps')
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
                lstm_hidden_size=256,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            verbose=1,
            tensorboard_log='./logs_ss/',
            device='auto'
        )
        print('Starting fresh SS model')

    checkpoint_callback = CheckpointCallback(
        save_freq=12500,
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
