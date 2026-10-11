#!/usr/bin/env python3
"""Verify MG_ENV_VERSION=2 (coordinator.py) before any run uses it.

    python3 check_mg_v2.py        # must end with ALL CHECKS PASS

v2 = (a) new ground (a 128-px block not visited this episode) counts against
`stuck`, and (b) off-route time is booked and settled on return / at the end.
1. Version 1 (the default) = git HEAD's coordinator.py: identical rewards,
   observations and outcomes on every trajectory below.
2. The invariant still holds on v2: from every start point of the synthetic
   mini-act, the demo's own inputs clear on its exact frame and return
   exactly CLEAR_BONUS.
3. v2 changes when time is paid, never how much: every episode's undiscounted
   return equals the v1 accounting of the same steps (-frames + route gain,
   + rest of route + CLEAR_BONUS on a clear, - DEATH_PENALTY on a death), and
   off-route steps pay 0 until settled.
4. `stuck` on v2: standing still is still cut after ~STUCK_FRAMES (unless
   something kills Sonic first, as on v1); running
   back left (on the route, no gain) lasts at least as long as on v1 and
   longer whenever it keeps reaching new ground.
Trajectories: seeded random actions from 8 start points of Tyler's real AIZ1
(fixture off), hold-left / stand-still from 6, and the synthetic mini-act's
invariant (fixture on, MG_TEST_GOAL_X=7100).
"""
import json, os, subprocess, sys, tempfile
import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HOME, 'demos', 'play_20260930-153946.npz')
SYNTH = os.path.join(HOME, 'demos', '_synthetic_aiz1_goal7100.npz')


def record(out, version, synth):
    os.environ['MG_LOG'] = os.path.join(tempfile.gettempdir(), f'check_mg_{os.getpid()}.jsonl')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if version is not None:
        os.environ['MG_ENV_VERSION'] = version
    if synth:
        os.environ['MG_TEST_GOAL_X'] = '7100'
    import coordinator as C
    import demos
    shadow = {'gain': 0, 'finish': 0}
    up, fin = C.RouteProgress.update, C.RouteProgress.finish

    def update(self, *a):
        g = up(self, *a)
        shadow['gain'] += g
        return g

    def finish(self):
        g = fin(self)
        shadow['finish'] += g
        return g

    C.RouteProgress.update, C.RouteProgress.finish = update, finish
    res = dict(version=getattr(C, 'MG_ENV_VERSION', 'head'))

    def episode(env, key, k, policy, max_steps=100000):
        obs, _ = env.reset(options={'key': key, 'ckpt': k})
        rews, obs_sum, total, v1 = [], [int(obs.sum())], 0.0, 0.0
        booked = 0
        for m in range(max_steps):
            shadow['gain'] = shadow['finish'] = 0
            f0 = env.frames
            obs, r, term, trunc, info = env.step(policy(m))
            n = env.frames - f0
            rr = r / C.REWARD_SCALE
            out = info.get('mg_result', {}).get('outcome') if (term or trunc) else None
            v1 += -n + shadow['gain'] + (shadow['finish'] + C.CLEAR_BONUS if out == 'clear' else 0) \
                - (C.DEATH_PENALTY if out == 'death' else 0)
            booked += rr == 0 and not (term or trunc) and n > 0
            rews.append(round(rr, 6)); obs_sum.append(int(obs.sum())); total += rr
            if term or trunc:
                break
        r_ = info['mg_result']
        return dict(k=k, outcome=r_['outcome'], frames=r_['frames'], target=r_['target'],
                    ret=round(total, 6), v1=round(v1, 6), rews=rews, obs=obs_sum, booked=int(booked))

    if synth:
        actions, _ = demos.load_play(SYNTH)
        env = C.SonicCoordinator(demo_files=[SYNTH], acts=['AIZ1'])
        env.demo_info()
        seg = env._segs[0]
        res['invariant'] = []
        for k, ck in enumerate(seg['ckpts']):
            f = ck['frame']
            res['invariant'].append(episode(env, seg['key'], k,
                                            lambda m: int(actions[f + 1 + 4 * m]) if f + 1 + 4 * m < len(actions) else 0))
    else:
        env = C.SonicCoordinator(demo_files=[REAL], acts=['AIZ1'])
        env.demo_info()
        seg = env._segs[0]
        K = len(seg['ckpts'])
        rng = np.random.default_rng(5)
        res['random'] = [episode(env, seg['key'], k, lambda m, a=rng.integers(9, size=4000): int(a[m]), 3000)
                         for k in np.linspace(0, K - 1, 8).astype(int).tolist()]
        mids = np.linspace(K // 6, K - 2, 6).astype(int).tolist()
        res['left'] = [episode(env, seg['key'], k, lambda m: 1) for k in mids]
        res['still'] = [episode(env, seg['key'], k, lambda m: 0) for k in mids]
    json.dump(res, open(out, 'w'))


def main():
    tmp = tempfile.mkdtemp(prefix='check_mg_')
    old = os.path.join(tmp, 'old')
    os.mkdir(old)
    for name in os.listdir(HOME):
        if name != 'coordinator.py':
            os.symlink(os.path.join(HOME, name), os.path.join(old, name))
    with open(os.path.join(old, 'coordinator.py'), 'wb') as f:
        f.write(subprocess.run(['git', 'show', 'HEAD:coordinator.py'], cwd=HOME,
                               check=True, capture_output=True).stdout)
    got = {}
    for tag, where, version in (('head', old, None), ('v1', HOME, None), ('v2', HOME, '2')):
        for synth in (False, True):
            out = os.path.join(tmp, f'{tag}_{int(synth)}.json')
            print(f'recording {tag} {"synthetic" if synth else "real AIZ1"} ...', flush=True)
            subprocess.run([sys.executable, '-c',
                            f'import sys; sys.path.insert(0, {where!r}); sys.argv=["x"]; '
                            f'import importlib.util as u; s=u.spec_from_file_location("cm", {__file__!r}); '
                            f'm=u.module_from_spec(s); s.loader.exec_module(m); m.record({out!r}, {version!r}, {synth!r})'],
                           cwd=where, check=True)
            got[tag, synth] = json.load(open(out))
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f'  {"ok  " if cond else "FAIL"} {msg}')

    strip = lambda d: {k: v for k, v in d.items() if k != 'version'}
    check(strip(got['head', False]) == strip(got['v1', False]) and strip(got['head', True]) == strip(got['v1', True]),
          'version 1 = git HEAD: identical rewards, observations and outcomes on every trajectory')
    inv = got['v2', True]['invariant']
    bad = [(e['k'], e['outcome'], e['frames'], e['target'], e['ret']) for e in inv
           if not (e['outcome'] == 'clear' and e['frames'] == e['target'] and abs(e['ret'] - 600) < 1e-6)]
    check(not bad, f'v2 invariant: {len(inv)} start points clear on the demo frame with return 600' +
          (f' — MISMATCHES {bad}' if bad else ''))
    eps = [e for part in ('random', 'left', 'still') for e in got['v2', False][part]] + inv
    off = [(e['k'], e['ret'], e['v1']) for e in eps if abs(e['ret'] - e['v1']) > 1e-6]
    booked = sum(e['booked'] for e in eps)
    check(not off and booked > 0, f'v2 returns = v1 accounting on all {len(eps)} episodes; '
          f'{booked} off-route steps paid 0 until settled' + (f' — DIFFER {off[:5]}' if off else ''))
    v1eps = [e for part in ('random', 'left', 'still') for e in got['v1', False][part]]
    check(all(abs(e['ret'] - e['v1']) < 1e-6 for e in v1eps), 'v1 returns = v1 accounting (the shadow sum is right)')
    still1, still2 = got['v1', False]['still'], got['v2', False]['still']
    # some start points kill a Sonic who stands still (deaths at 555 / 197 frames on every version)
    check(all((b['outcome'] == 'stuck' and 600 <= b['frames'] <= 800)
              or (b['outcome'] == a['outcome'] == 'death' and b['frames'] == a['frames'])
              for a, b in zip(still1, still2)),
          f'v2 standing still: cut as stuck after ~STUCK_FRAMES unless killed first, deaths as on v1: '
          f'{[(e["outcome"], e["frames"]) for e in still2]}')
    l1, l2 = got['v1', False]['left'], got['v2', False]['left']
    pairs = [(a['k'], a['outcome'], a['frames'], b['outcome'], b['frames']) for a, b in zip(l1, l2)]
    check(all(b['frames'] >= a['frames'] for a, b in zip(l1, l2)),
          f'v2 hold-left lasts >= v1 from every start point (start, v1 outcome/frames, v2 outcome/frames): {pairs}')
    r1, r2 = got['v1', False]['random'], got['v2', False]['random']
    print(f'  info: random play outcomes v1 {[(e["outcome"], e["frames"]) for e in r1]}')
    print(f'        random play outcomes v2 {[(e["outcome"], e["frames"]) for e in r2]}')
    print('\nALL CHECKS PASS' if ok else '\nCHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
