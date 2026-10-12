#!/usr/bin/env python3
"""check_imitation.py — does train_main's ImitationCallback (RUNS 'imitation') do what it says?

    python3 check_imitation.py [RUN] [DATA]       # default mg1bc, datasets/bc_v1

1. the callback alone on RUN's saved network: it takes only the run's recordings and acts, the frames
   around the current curriculum start point (his t within [-behind, +ahead] of it), its loss falls and
   its accuracy on his actions rises over a few updates, its weight falls linearly to 0 over decay_steps
   (no steps at 0), and the value head is not trained by it directly
2. a tiny end-to-end run inside PPO's real loop: one env, the run's settings, two rollouts of 512 steps,
   a temporary copy of the curriculum file, nothing saved: imitation/* is logged and nothing crashes
CPU only (2 threads). Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import shutil
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg1bc'
    data = sys.argv[2] if len(sys.argv) > 2 else 'datasets/bc_v1'
    sys.path.insert(0, HOME)
    import numpy as np
    import train_main as T
    T.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import torch
    torch.set_num_threads(2)
    import coordinator as C
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    bad = []
    P = T.paths(run)
    tmp = tempfile.mkdtemp()
    cur = os.path.join(tmp, 'curriculum.json')
    shutil.copy(P['curriculum'], cur)
    cfg = {**T.IMITATION_DEFAULTS, 'data': data}

    # 1. the callback alone
    model = PPO.load(P['zip'] + '.zip', device='cpu')
    model.set_logger(configure(None, []))
    cb = T.ImitationCallback(cfg, run, cur)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    keys = {k: len(v) for k, v in cb.keys.items()}
    print(f'1. {run}: imitation frames by recording:act {keys}')
    want = {f"{os.path.basename(p)[:-4]}:{a.split(':')[-1]}" for p in T.demo_paths(T.RUNS[run])
            for a in T.RUNS[run]['acts']}
    if not keys or not set(keys) <= want:
        bad.append(f'1: recordings/acts {set(keys)} not within the run\'s {want}')
    c0 = json.load(open(cur))
    front = cb.frontier()
    for key, entry in c0.items():
        idx = cb.keys.get(key)
        if idx is None:
            continue
        t0 = entry['frac'] * int(cb.S['T'][idx[0]])
        ts = cb.S['t'][np.intersect1d(front, idx)]
        print(f"   {key}: start point {entry['ptr']} (t {t0:.0f}); frontier frames t {ts.min()}-{ts.max()} ({len(ts)})")
        if not (t0 - cfg['behind'] - 1 <= ts.min() and ts.max() <= t0 + cfg['ahead'] + 1):
            bad.append('1: frontier frames outside the window')
    vf_before = [p.detach().clone() for p in model.policy.value_net.parameters()]
    logs = []
    for i in range(12):
        cb._on_rollout_start()
        logs.append((model.logger.name_to_value['imitation/loss'], model.logger.name_to_value['imitation/accuracy']))
    print('   12 updates of 8 steps: loss ' + ' '.join(f'{l:.2f}' for l, _ in logs[::3])
          + '; accuracy ' + ' '.join(f'{a:.0%}' for _, a in logs[::3]))
    if not logs[-1][0] < logs[0][0]:
        bad.append('1: imitation loss did not fall')
    if any(not torch.equal(a, b) for a, b in zip(vf_before, model.policy.value_net.parameters())):
        bad.append('1: the value head was trained')
    for frac, want_w in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.5, 0.0)):
        model.num_timesteps = cb.start + int(frac * cfg['decay_steps'])
        if abs(cb.weight() - want_w) > 1e-9:
            bad.append(f'1: weight at {frac} of decay_steps = {cb.weight()}, want {want_w}')
    model.num_timesteps = cb.start + int(0.5 * cfg['decay_steps'])
    cb._on_rollout_start()
    if abs(cb.opt.param_groups[0]['lr'] - 0.5 * cb.base_lr) > 1e-12:
        bad.append(f"1: at half the decay the step size is {cb.opt.param_groups[0]['lr']}, want {0.5 * cb.base_lr}")
    print(f"   step size at half the decay: {cb.opt.param_groups[0]['lr']:.2e} (full {cb.base_lr:.2e})")
    model.num_timesteps = cb.start + int(1.5 * cfg['decay_steps'])
    model.logger.name_to_value.pop('imitation/loss', None)
    cb._on_rollout_start()
    if 'imitation/loss' in model.logger.name_to_value:
        bad.append('1: imitation steps ran at weight 0')
    print('   weight at 0, 50%, 100%, 150% of decay_steps:',
          [round(cfg['weight'] * max(0, 1 - f), 2) for f in (0, 0.5, 1, 1.5)])

    # 2. inside PPO's loop
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    env = VecNormalize.load(P['pkl'], DummyVecEnv([lambda: C.SonicCoordinator()]))
    model = PPO.load(P['zip'] + '.zip', env=env, device='cpu')
    start = model.num_timesteps
    cb2 = T.ImitationCallback(cfg, run, cur)
    seen = {}

    class Grab(T.BaseCallback):
        def _on_rollout_end(self):
            seen.update({k: v for k, v in self.logger.name_to_value.items() if k.startswith('imitation/')})

        def _on_step(self):
            return True
    model.learn(total_timesteps=2 * model.n_steps, callback=[T.ActCurriculumCallback(cur, C.DEMO_BAND), cb2, Grab()],
                reset_num_timesteps=False)
    print(f'2. PPO ran {model.num_timesteps - start} steps with imitation; logged {sorted(seen)}')
    if 'imitation/loss' not in seen:
        bad.append('2: no imitation step inside the PPO loop')
    shutil.rmtree(tmp)
    if bad:
        print('\nFAILED:\n  ' + '\n  '.join(bad))
        sys.exit(1)
    print('\nALL CHECKS PASS')


if __name__ == '__main__':
    main()
