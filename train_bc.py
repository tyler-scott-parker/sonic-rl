#!/usr/bin/env python3
"""train_bc.py — pretrain the main-game agent to copy Tyler (build_bc_dataset.py's pairs), then
reinforcement learning starts from it (RUNS 'init').

    python3 train_bc.py OUT_RUN [--init mg1] [--data datasets/bc_v1] [--epochs 3] [--threads 2]
    python3 train_bc.py bc1                     # -> main_bc1.zip + main_bc1_vecnormalize.pkl

Starts from a run's PPO model (its network has learned to see the game; default mg1) and trains the
same ActorCriticCnnPolicy on: the log-likelihood of his action (each sample weighted by how many of its
4 frames agree with the label, /4), and the value of his run from there: 600 frames (the clear bonus)
discounted by gamma per agent step to the clear, in the run's reward units (REWARD_SCALE / the
VecNormalize return std), since on his own route each step's reward is 0 (it pays the route back) and
the clear pays CLEAR_BONUS. Held out: 1-second blocks (60 frames), 1 in 20 of each act: accuracy on
those = how well it copies him on frames it never trained on. Saves a PPO zip RUNS can init from.
CPU: 2 threads (nice beside a game) or GPU.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time

HOME = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--init', default='mg1')
    ap.add_argument('--data', default=os.path.join(HOME, 'datasets', 'bc_v1'))
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--soft', type=int, default=0, metavar='W',
                    help='target = his rounded actions over frames i-W..i+W (timing-tolerant), not frame i alone')
    ap.add_argument('--jump-weight', type=float, default=1.0, help='weight of samples whose label jumps')
    ap.add_argument('--exclude-acts', default='', metavar='ACT,ACT',
                    help="leave these acts out (e.g. one RL already plays better than his copy: a "
                         "copying pass over AIZ1 erased mg1bc's from-start AIZ1 clears, 5/20 -> 0/20)")
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    import train_main
    from stable_baselines3 import PPO
    import coordinator as C

    src = train_main.paths(a.init)
    dst = train_main.paths(a.out)
    if os.path.exists(dst['zip'] + '.zip'):
        sys.exit(f"{dst['zip']}.zip exists: pick another name")
    model = PPO.load(src['zip'] + '.zip', device=a.device)
    pol = model.policy
    vn = pickle.load(open(src['pkl'], 'rb'))
    ret_std = float(np.sqrt(vn.ret_rms.var + vn.epsilon))
    gamma = model.gamma
    meta = json.load(open(os.path.join(a.data, 'meta.json')))
    S = dict(np.load(os.path.join(a.data, 'samples.npz')))
    frames = [np.load(os.path.join(a.data, f"frames_{d['id']}.npy"), mmap_mode='r') for d in meta['demos']]
    n = len(S['action'])
    steps_left = (S['T'] - S['t']) / C.FRAME_SKIP
    v_target = (C.CLEAR_BONUS * gamma ** steps_left * C.REWARD_SCALE / ret_std).astype(np.float32)
    weight = (S['agree'] / C.FRAME_SKIP).astype(np.float32)
    jumps = np.array([C.ACTION_BUTTONS[k][0] == 1 for k in range(len(C.ACTION_NAMES))])
    weight[jumps[S['action']]] *= a.jump_weight
    K = len(C.ACTION_NAMES)
    soft = None
    if a.soft:                                              # neighbouring frames' labels, same act + demo
        soft = np.zeros((n, K), dtype=np.float32)
        for d in range(-a.soft, a.soft + 1):
            j = np.clip(np.arange(n) + d, 0, n - 1)
            same = (S['demo'][j] == S['demo']) & (S['act'][j] == S['act']) & (np.abs(S['frame'][j] - S['frame'] - d) == 0)
            src_lbl = np.where(same, S['action'][j], S['action'])
            soft[np.arange(n), src_lbl] += 1.0
        soft /= soft.sum(1, keepdims=True)
    held = ((S['t'] // 60) % 20 == 7)                       # 1-second blocks, 1 in 20 of each act
    out = np.isin(S['act'], [meta['acts'].index(x) for x in a.exclude_acts.split(',') if x in meta['acts']])
    tr, va = np.flatnonzero(~held & ~out), np.flatnonzero(held & ~out)
    offs = np.arange(-(C.STACK - 1) * C.FRAME_SKIP, 1, C.FRAME_SKIP)          # -12, -8, -4, 0
    print(f'{n:,} samples ({len(tr):,} train, {len(va):,} held out; {int(out.sum()):,} left out: '
          f'{a.exclude_acts or "none"}) from {len(meta["demos"])} recordings, '
          f'{len(meta["acts"])} acts; init {a.init} ({model.num_timesteps:,} steps); value target '
          f'600 x gamma^steps x {C.REWARD_SCALE} / {ret_std:.3f}', flush=True)

    def batch(idx):
        obs = np.empty((len(idx), C.STACK, C.OBS_SIZE, C.OBS_SIZE), dtype=np.uint8)
        for j, s in enumerate(idx):
            fr = frames[S['demo'][s]]
            obs[j] = fr[np.clip(S['frame'][s] + offs, 0, None)]
        return (torch.as_tensor(obs, device=a.device).float(),
                torch.as_tensor(S['action'][idx], device=a.device),
                torch.as_tensor(weight[idx], device=a.device),
                torch.as_tensor(v_target[idx], device=a.device),
                None if soft is None else torch.as_tensor(soft[idx], device=a.device))

    def evaluate(idx, cap=20000):
        pol.eval()
        idx = idx[:cap]
        hit, per = 0, np.zeros((len(C.ACTION_NAMES), 2))
        with torch.no_grad():
            for b in range(0, len(idx), 512):
                obs, act, _, _, _ = batch(idx[b:b + 512])
                dist = pol.get_distribution(obs)
                pred = dist.distribution.probs.argmax(-1).cpu().numpy()
                act = act.cpu().numpy()
                hit += (pred == act).sum()
                for k in range(len(C.ACTION_NAMES)):
                    per[k] += [((pred == act) & (act == k)).sum(), (act == k).sum()]
        pol.train()
        return hit / len(idx), {C.ACTION_NAMES[k]: f'{per[k, 0] / per[k, 1]:.0%} of {int(per[k, 1])}'
                                for k in range(len(C.ACTION_NAMES)) if per[k, 1]}

    acc0, _ = evaluate(va)
    print(f'before: held-out accuracy {acc0:.1%} ({a.init} as it is)', flush=True)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    total = int(a.epochs * len(tr) / a.batch)
    t0, run = time.time(), [0.0, 0.0, 0]
    for it in range(1, total + 1):
        idx = rng.choice(tr, a.batch, replace=False)
        obs, act, w, vt, sl = batch(idx)
        values, logp, _ = pol.evaluate_actions(obs, act)
        if sl is not None:                                  # cross-entropy with the soft target
            logits = pol.get_distribution(obs).distribution.logits
            logp = (sl * torch.log_softmax(logits, -1)).sum(-1)
        pi_loss = -(w * logp).sum() / w.sum()
        v_loss = ((values.flatten() - vt) ** 2).mean()
        loss = pi_loss + a.vf * v_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
        opt.step()
        run = [run[0] + pi_loss.item(), run[1] + v_loss.item(), run[2] + 1]
        if it % 500 == 0 or it == total:
            acc, per = evaluate(va, cap=5000 if it < total else 10 ** 9)
            print(f'  {it:6d}/{total} ({it * a.batch / len(tr):.2f} epochs): policy loss {run[0] / run[2]:.3f}, '
                  f'value loss {run[1] / run[2]:.4f}, held-out accuracy {acc:.1%} '
                  f'[{(time.time() - t0) / 60:.0f} min]', flush=True)
            run = [0.0, 0.0, 0]
    acc, per = evaluate(va, cap=10 ** 9)
    print(f'after: held-out accuracy {acc:.1%} (before {acc0:.1%}); per action: {per}', flush=True)
    model.num_timesteps = 0
    model.save(dst['zip'])
    shutil.copy(src['pkl'], dst['pkl'])
    print(f"saved {dst['zip']}.zip (+ {dst['pkl']}): RUNS init='{a.out}'")


if __name__ == '__main__':
    main()
