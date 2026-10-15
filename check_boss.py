#!/usr/bin/env python3
"""Verify the boss-hit features (coordinator.py, "Boss hits") before any run uses them.

    python3 check_boss.py        # must end with ALL CHECKS PASS (CPU only; ~10 min)

1. The env's BossTracker on Tyler's whole-game recording up to MGZ2's clear,
   updated every agent step (4 frames) as the env does: exactly AIZ1 6, AIZ2 8,
   HCZ1 6, HCZ2 8, MGZ1 6, MGZ2 8 hits, each act's last within a step of its
   clear, and no other hit anywhere.
2. Start points inside a fight know their boss (the tracker's memory rides
   with each start point): the bar at reset gives (hits left, 6 or 8).
3. MG_BOSS_PAY changes when time is paid, never how much: the mg2 agent plays
   episodes from start points around each act's boss with pay on (A, bar on);
   the same actions replayed with pay 0 (B, bar on) give the same
   observations, outcomes and, per episode, exactly the same undiscounted
   return; per-step rewards differ only on hit steps and the settling steps;
   a clear returns exactly (his frames - agent frames + CLEAR_BONUS).
4. `stuck`: the same actions with tracking off (C) — reported, not asserted:
   how many episodes the old rule cut as stuck where A went on.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
PAY = 120.0
EXPECT = {'AIZ1': 6, 'AIZ2': 8, 'HCZ1': 6, 'HCZ2': 8, 'MGZ1': 6, 'MGZ2': 8}
FULL = os.path.join(HOME, 'demos', 'play_20261010-202923.npz')


def check_replay():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import coordinator as C
    import demos
    actions, meta = demos.load_play(FULL)
    table = meta['action_table']
    acts = [(a['name'], a['start'], a['clear']) for a in meta['acts'] if a['name'] in EXPECT]
    end = max(c for _, _, c in acts) + 240
    env = C.make_retro()
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    bt, got, stray = C.BossTracker(), {a: [] for a in EXPECT}, []
    for i, a in enumerate(actions[:end]):
        _, info = C.emu_frame(env, table[a])
        if i % C.FRAME_SKIP:
            continue
        h = bt.update(env, info, C.FRAME_SKIP)
        if h:
            act = next((n for n, s, c in acts if s <= i <= c + C.FRAME_SKIP), None)
            (got[act] if act else stray).extend([i] * h)
    ok = True
    for name, s, c in acts:
        n, last = len(got[name]), (got[name][-1] - c if got[name] else None)
        good = n == EXPECT[name] and last is not None and 0 <= last < C.FRAME_SKIP
        ok &= good
        print(f"  {name}: {n} hits (expect {EXPECT[name]}), last {last:+d} frames from the clear"
              f"{'' if good else '   <-- FAIL'}")
    print(f'  hits outside these acts: {len(stray)}' + ('' if not stray else '   <-- FAIL'))
    return ok and not stray


def record(mode, out, plan):
    """One process: build mg2's acts with this mode's flags and play the plan."""
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env('mg2')
    os.environ['MG_LOG'] = os.path.join(tempfile.gettempdir(), f'check_boss_{os.getpid()}.jsonl')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MG_BOSS_BAR'] = '0' if mode == 'C' else '1'
    os.environ['MG_BOSS_PAY'] = str(PAY) if mode == 'A' else '0'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    env = C.SonicCoordinator(acts=train_main.RUNS['mg2']['acts'], demo_prob=0.0)
    env.demo_info()
    cur = json.load(open(train_main.paths('mg2')['curriculum']))
    if mode == 'A':
        from stable_baselines3 import PPO
        model = PPO.load(train_main.paths('mg2')['zip'] + '.zip', device='cpu')
        torch.manual_seed(0)
        plan = []
        for seg in env._segs:
            ptr, last = cur[seg['key']]['ptr'], len(seg['ckpts']) - 1
            for k in sorted({max(0, ptr - 10), max(0, ptr - 5), ptr, last - 1}):
                plan.append(dict(key=seg['key'], k=k))
    eps = []
    for ep in plan:
        obs, _ = env.reset(options={'key': ep['key'], 'ckpt': ep['k']})
        bar0 = env._bt.bar() if env._bt is not None else None
        acts, rews, hits, obs_sum = [], [], [], []
        for t in range(100000):
            if mode == 'A':
                a = int(model.predict(obs, deterministic=False)[0])
            else:
                if t >= len(ep['actions']):
                    break
                a = ep['actions'][t]
            n0 = env._boss_hits
            obs, r, term, trunc, info = env.step(a)
            acts.append(a)
            rews.append(float(r) / C.REWARD_SCALE)
            hits.append(env._boss_hits - n0)
            obs_sum.append(hashlib.md5(obs.tobytes()).hexdigest()[:12])
            if term or trunc:
                break
        res = info.get('mg_result', {})
        eps.append(dict(key=ep['key'], k=ep['k'], actions=acts, rews=rews, hits=hits, obs=obs_sum,
                        ret=env.ret, outcome=res.get('outcome'), frames=env.frames, target=env._target,
                        bar0=bar0, stages=len(env._stages), ended=bool(term or trunc)))
        print(f"  [{mode}] {ep['key'].split(':')[-1]} start {ep['k']}: {res.get('outcome')} after {env.frames} "
              f"frames, {sum(hits)} hits, return {env.ret:+.0f}", flush=True)
    json.dump(eps, open(out, 'w'))


def run_mode(mode, plan_path, out):
    cmd = [sys.executable, os.path.abspath(__file__), '--record', mode, '--out', out]
    if plan_path:
        cmd += ['--plan', plan_path]
    subprocess.run(cmd, check=True, cwd=HOME)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record')
    ap.add_argument('--out')
    ap.add_argument('--plan')
    a = ap.parse_args()
    if a.record:
        plan = json.load(open(a.plan)) if a.plan else None
        return record(a.record, a.out, plan)
    sys.path.insert(0, HOME)
    results = {}
    print('1. BossTracker on his recording, every agent step:', flush=True)
    results[1] = check_replay()

    tmp = tempfile.mkdtemp(prefix='check_boss_')
    A = run_mode('A', None, os.path.join(tmp, 'A.json'))
    plan = [dict(key=e['key'], k=e['k'], actions=e['actions']) for e in A]
    json.dump(plan, open(os.path.join(tmp, 'plan.json'), 'w'))
    B = run_mode('B', os.path.join(tmp, 'plan.json'), os.path.join(tmp, 'B.json'))
    Cc = run_mode('C', os.path.join(tmp, 'plan.json'), os.path.join(tmp, 'C.json'))

    print('\n2. start points inside fights know their boss (bar at reset):')
    ok2, seen = True, 0
    for e in A:
        if e['bar0'] is not None:
            seen += 1
            left, total = e['bar0']
            good = total in (6, 8) and 0 < left <= total
            ok2 &= good
            print(f"  {e['key'].split(':')[-1]} start {e['k']}: {left} of {total} left" + ('' if good else '   <-- FAIL'))
    results[2] = ok2 and seen > 0

    print('\n3. pay on (A) vs off (B), same actions:')
    ok3, hits_total = True, 0
    from coordinator import CLEAR_BONUS                     # noqa: E402 (flags don't matter here)
    for ea, eb in zip(A, B):
        tag = f"{ea['key'].split(':')[-1]} start {ea['k']}"
        hits_total += sum(ea['hits'])
        same = (ea['obs'] == eb['obs'] and ea['outcome'] == eb['outcome'] and ea['frames'] == eb['frames']
                and ea['hits'] == eb['hits'])
        dret = ea['ret'] - eb['ret']
        diff = [i for i, (x, y) in enumerate(zip(ea['rews'], eb['rews'])) if abs(x - y) > 1e-6]
        hit_steps = {i for i, h in enumerate(ea['hits']) if h}
        extra = [i for i in diff if i not in hit_steps]
        clear_ok = True
        if ea['outcome'] == 'clear' and not ea['stages']:
            clear_ok = abs(ea['ret'] - (ea['target'] - ea['frames'] + CLEAR_BONUS)) < 1e-6
        good = same and abs(dret) < 1e-6 and clear_ok and len(extra) <= 2 * max(1, len(hit_steps))
        ok3 &= good
        print(f"  {tag}: {sum(ea['hits'])} hits, {ea['outcome']}, return {ea['ret']:+.0f} vs {eb['ret']:+.0f}; "
              f"steps differing {len(diff)} (hit steps {len(hit_steps)})"
              + ('' if clear_ok else '; CLEAR RETURN WRONG') + ('' if same else '; TRAJECTORY DIFFERS')
              + ('' if good else '   <-- FAIL'))
    print(f'  hits in all: {hits_total}' + ('' if hits_total else '   <-- FAIL (nothing tested)'))
    results[3] = ok3 and hits_total > 0

    print('\n4. stuck: tracking on (A) vs off (C), same actions (report only):')
    for ea, ec in zip(A, Cc):
        if ea['outcome'] != ec['outcome'] or ea['frames'] != ec['frames']:
            print(f"  {ea['key'].split(':')[-1]} start {ea['k']}: on {ea['outcome']} at {ea['frames']}, "
                  f"off {ec['outcome']} at {ec['frames']}")
    print('  (others identical)')

    print()
    for k, v in results.items():
        print(f'check {k}: {"pass" if v else "FAIL"}')
    print('ALL CHECKS PASS' if all(results.values()) else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
