"""
train_ss_multistage_v2.py — Step-2 scaling test.

Step 1 (2 stages, position dropped) VALIDATED: transfer ratio 0.06 -> 0.75.
Step 2 asks: does that generalization HOLD as we scale to 5 stages, AND does the
skill reach boards it has NEVER trained on (true zero-shot transfer)?

TRAIN on 5 stages: 1,2,3,4,7   (blue counts 102/127/118/115/60)
HOLD OUT 2 stages: 5,6         (blue counts 58/128 — span small+large; never trained,
                                only evaluated at the end with zeroshot_test.py)

Position stays DROPPED (SS_DROP_POSITION=1). Do NOT add coarse position here —
Step 2 changes ONLY the stage count so the result is interpretable. Option B is a
separate controlled run if depth turns out too shallow.

Fresh from random weights. New checkpoint names (multistage_v2) — never touches the
v1 probe or the sonic_ss_model_v2 specialist.

LAUNCH:
  cd ~/sonic-rl && source venv/bin/activate
  SS_DROP_POSITION=1 SS_STAGES=1,2,3,4,7 SS_RESTART_PROB=0 \
    nohup python3 train_ss_multistage_v2.py > logs/ss_multistage_v2.log 2>&1 &

READ (partway + at the end):
  python3 stage_split.py 8000 --curve     # do the 5 TRAINED stages climb together?
  python3 zeroshot_test.py                 # do the 2 HELD-OUT stages (5,6) transfer?
"""

import os
import glob
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                CallbackList)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("pip install sb3-contrib --break-system-packages")

from special_stage_env import SpecialStageEnv

# ── Config (NEW names — does not collide with the v2 specialist) ──────────────
NUM_ENVS    = 8
PKL         = 'sonic_ss_multistage_v2_vecnormalize.pkl'
ZIP         = 'sonic_ss_multistage_v2'
SAVE_FREQ   = 500_000
TOTAL_STEPS = 50_000_000          # Step 2: 5 stages need more than the 25M 2-stage probe; open-ended (autosaves, resumes) — ride the --curve until the data is convincing
CKPT_DIR    = './checkpoints_ss_multistage_v2/'

# Surface the probe config in the log so the run is self-documenting.
print(f"SS_DROP_POSITION={os.environ.get('SS_DROP_POSITION','0')}  "
      f"SS_STAGES={os.environ.get('SS_STAGES','(unset)')}  "
      f"SS_RESTART_PROB={os.environ.get('SS_RESTART_PROB','0.25')}")


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

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Fresh VecNormalize — do NOT load the v2 pkl (its obs stats are for the
    # position-encoded specialist; this run has a different observation).
    if os.path.exists(PKL):
        env = VecNormalize.load(PKL, env)
        env.training = True
        print(f'Resumed VecNormalize from {PKL}')
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0)
        print('Created fresh VecNormalize (multistage v1)')

    # Resume THIS run if it exists; otherwise start from random weights.
    if os.path.exists(f'{ZIP}.zip'):
        model = RecurrentPPO.load(ZIP, env=env, device='auto')
        print(f'Resuming multistage v1 at {model.num_timesteps} steps')
    else:
        model = RecurrentPPO(
            policy='MultiInputLstmPolicy',
            env=env,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            learning_rate=3e-4,     # fresh run: warm lr (the 5e-5 was a cooled
                                    # value for the mature specialist)
            ent_coef=0.01,
            clip_range=0.1,
            gamma=0.997,            # keep the long endgame horizon from the fix
            gae_lambda=0.95,
            policy_kwargs=dict(
                lstm_hidden_size=512,
                n_lstm_layers=2,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
            verbose=1,
            tensorboard_log='./logs_ss/',
            device='auto',
        )
        print('Starting FRESH multistage v1 model (random weights)')

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path=CKPT_DIR,
        name_prefix='sonic_ss_ms_v2', verbose=1)
    callbacks = CallbackList([
        checkpoint_callback,
        AutoSaveCallback(save_freq=SAVE_FREQ),
        PruneCheckpointsCallback(checkpoint_dir=CKPT_DIR, keep=5),
    ])

    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks,
                reset_num_timesteps=False)
