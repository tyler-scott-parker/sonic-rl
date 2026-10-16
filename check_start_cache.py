#!/usr/bin/env python3
"""check_start_cache.py — do start points from start_cache behave exactly like freshly built ones?

    python3 check_start_cache.py mg2          # main game (train_main RUNS entry)
    python3 check_start_cache.py d10          # special stages (train_ss_multistage RUNS entry)

Each part runs in its own process (one emulator per process), with the run's own settings:
1. the trainer's start-up: mg2 coordinator.prebuild(); d10 replay_check + prebuild_demos for the
   demos not in the cache. Writes the REAL start_cache (so the run's next start is fast); a second
   call must report it cached
2. fresh: the cache pointed at an empty folder, the start points replayed as before start_cache
3. cached: the real cache, nothing replayed (checked: the build takes under 20 s)
2 and 3 must give the same start points: a canonical digest of every observation stack, route,
fight, clip and bookkeeping field, and for the emulator snapshots, the same emulation: every 10th
start point restored and played 240 frames of seeded random buttons, screen + RAM every frame.
(Snapshot bytes themselves differ between any two processes: Genesis Plus GX writes ~145 KB of
state into a fixed 1 MB buffer whose tail is leftover memory, and its YM2612 channels hold host
pointers that loading recomputes.) And, from the same seed after the build, the same 3,000 steps
of seeded random play with the run's curriculum pointers: every observation, reward, end, position.
Nothing is logged (episode logs go to /dev/null; d10's route saves off). CPU only.
Ends with ALL CHECKS PASS or the failures.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HOME = os.path.dirname(os.path.abspath(__file__))
STEPS = 3000
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what, flush=True)
    if not ok:
        bad.append(what)


def digest(obj, h=None):
    """A canonical sha1 of nested dicts/lists/tuples/sets/arrays/bytes/scalars."""
    import numpy as np
    top = h is None
    h = h or hashlib.sha1()
    if isinstance(obj, dict):
        h.update(b'{')
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            digest(obj[k], h)
        h.update(b'}')
    elif isinstance(obj, (list, tuple)):
        h.update(b'[' if isinstance(obj, list) else b'(')
        for x in obj:
            digest(x, h)
        h.update(b']')
    elif isinstance(obj, (set, frozenset)):
        h.update(b's' + repr(sorted(obj, key=repr)).encode())
    elif isinstance(obj, np.ndarray):
        h.update(f'a{obj.dtype}{obj.shape}'.encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (bytes, bytearray)):
        h.update(b'b' + bytes(obj))
    elif isinstance(obj, np.generic):
        h.update(f'n{obj.dtype}{obj.item()!r}'.encode())
    else:
        h.update(f'{type(obj).__name__}{obj!r}'.encode())
    return h.hexdigest() if top else None


def strip(obj):
    """obj without its emulator snapshots ('state' bytes), for the digest."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != 'state'}
    if isinstance(obj, (list, tuple)):
        return type(obj)(strip(x) for x in obj)
    return obj


def emulate(base_env, states, buttons):
    """Per snapshot: sha1 of screen + RAM over 240 frames of seeded random buttons from it."""
    import zlib
    import numpy as np
    rng = np.random.default_rng(0)
    out = []
    for st in states:
        base_env.em.set_state(zlib.decompress(st))
        h = hashlib.sha1()
        for _ in range(240):
            base_env.em.set_button_mask(np.asarray(buttons[int(rng.integers(len(buttons)))], np.uint8), 0)
            base_env.em.step()
            h.update(base_env.em.get_screen().tobytes())
            h.update(base_env.get_ram().tobytes())
        out.append(h.hexdigest())
    return out


def play(env, ptrs, info_keys):
    """3,000 seeded random steps from reset; per step a sha1 of obs + reward + ends + some info."""
    import random
    import numpy as np
    random.seed(0)
    np.random.seed(0)
    env.set_demo_pointers(ptrs)
    rng = np.random.default_rng(0)
    h = hashlib.sha1()
    obs, _ = env.reset(seed=0)
    eps = 0
    for i in range(STEPS):
        o = obs if not isinstance(obs, dict) else [obs[k] for k in sorted(obs)]
        for x in (o if isinstance(o, list) else [o]):
            h.update(np.ascontiguousarray(x).tobytes())
        obs, r, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
        h.update(repr((round(float(r), 6), bool(term), bool(trunc),
                       [info.get(k) for k in info_keys])).encode())
        if term or trunc:
            eps += 1
            obs, _ = env.reset()
    return h.hexdigest(), eps


# ── the parts, each in its own process ─────────────────────────────────────────

def setup(run):
    sys.path.insert(0, HOME)
    os.chdir(HOME)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if run.startswith('mg'):
        import train_main
        train_main.configure_env(run)
        os.environ['MG_LOG'] = os.devnull           # not in the cache key
        import coordinator
        coordinator.LOG_FILE = os.devnull
        return coordinator
    src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
    sys.argv = ['train_ss_multistage.py', run, '--resume']
    g = {'__name__': 'ss_head', '__file__': os.path.join(HOME, 'train_ss_multistage.py')}
    exec(compile(src[:src.index("if __name__ == '__main__':")], 'train_ss_multistage.py', 'exec'), g)
    import special_stage_env as sse
    sse.ROUTE_DIR = ''                                 # no route saves from test play
    return sse, g


def part(run, which, cache_dir, ptr_file):
    import start_cache
    out = {}
    if run.startswith('mg'):
        C = setup(run)
        if which == 'prebuild':
            start_cache.DIR = cache_dir
            out['note1'] = C.prebuild()
            out['note2'] = C.prebuild()
            return out
        start_cache.DIR = cache_dir
        env = C.SonicCoordinator()
        t = time.time()
        env.demo_info()
        out['build_s'] = time.time() - t
        out['digest'] = digest(strip([env._segs, env._clips]))
        out['points'] = sum(len(s['ckpts']) for s in env._segs)
        states = [c['state'] for s in env._segs for c in s['ckpts']]
        states += [p['state'] for cl in env._clips.values() for c in cl for p in c['pts']]
        out['emu'] = emulate(env.base_env, states[::10], C.ACTION_BUTTONS)
        out['play'], out['eps'] = play(env, json.load(open(ptr_file)), ['x', 'y', 'zone', 'act'])
        env.close()
        return out
    sse, g = setup(run)
    demos = g['DEMOS']
    if which == 'prebuild':
        import demos as demo_io
        start_cache.DIR = cache_dir
        todo = [d for d in demos if not sse.demo_cached(d)]
        out['checks'] = [demo_io.replay_check(d) for d in todo]
        out['took'] = sse.prebuild_demos(todo) if todo and all(ok for ok, _ in out['checks']) else 0
        out['todo'] = len(todo)
        out['left'] = sum(not sse.demo_cached(d) for d in demos)
        return out
    start_cache.DIR = cache_dir
    env = sse.SpecialStageEnv()
    env._log_episode = lambda *a, **k: None
    t = time.time()
    env.demo_info()
    out['build_s'] = time.time() - t
    out['digest'] = digest(strip([env._demos, env._original_grids]))
    out['points'] = sum(len(d['ckpts']) for d in env._demos)
    states = [c['state'] for d in env._demos for c in d['ckpts']]
    out['emu'] = emulate(env.base_env, states[::10], sse.ACTION_BUTTONS)
    out['play'], out['eps'] = play(env, json.load(open(ptr_file)), ['ss_x', 'ss_y'])
    env.close()
    return out


def run_part(run, which, cache_dir, ptr_file=''):
    r = subprocess.run([sys.executable, '-u', __file__, '--part', run, which, cache_dir, ptr_file],
                       capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    if r.returncode or not lines:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f'part {which} failed (exit {r.returncode})')
    return json.loads(lines[-1][7:])


def main():
    if sys.argv[1:2] == ['--part']:
        run, which, cache_dir, ptr_file = sys.argv[2:6]
        print('RESULT ' + json.dumps(part(run, which, cache_dir, ptr_file)), flush=True)
        return 0
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    import start_cache
    real = start_cache.DIR
    print(f'1. trainer start-up ({run}), real cache {real}', flush=True)
    p = run_part(run, 'prebuild', real)
    if run.startswith('mg'):
        print(f"     first:  {p['note1']}\n     second: {p['note2']}")
        check(p['note2'].startswith('from start_cache'), 'a second start-up finds it cached')
    else:
        check(all(ok for ok, _ in p['checks']), f"replay_check passed the {p['todo']} uncached demo(s)")
        print(f"     {p['todo']} replayed in {p['took']:.0f} s")
        check(p['left'] == 0, 'every demo is cached afterwards')
    # the run's curriculum pointers, read once (a live trainer may move them between the parts)
    state = os.path.join(HOME, f'main_{run}_curriculum.json' if run.startswith('mg') else f'sonic_ss_multistage_{run}_demos.json')
    ptrs = {k: int(v['ptr']) for k, v in json.load(open(state)).items()}
    with tempfile.TemporaryDirectory() as tmp:
        ptr_file = os.path.join(tmp, 'ptrs.json')
        json.dump(ptrs, open(ptr_file, 'w'))
        empty = os.path.join(tmp, 'empty')
        os.makedirs(empty)
        print(f'2. fresh build (empty cache), pointers of {len(ptrs)} acts/boards from {os.path.basename(state)}', flush=True)
        fresh = run_part(run, 'play', empty, ptr_file)
        check(not os.listdir(empty), 'a worker-style build writes nothing to the cache')
        print(f"     {fresh['points']} start points in {fresh['build_s']:.0f} s; {fresh['eps']} episodes in {STEPS} steps")
        print('3. cached', flush=True)
        cached = run_part(run, 'play', real, ptr_file)
    print(f"     {cached['points']} start points in {cached['build_s']:.1f} s; {cached['eps']} episodes")
    check(cached['build_s'] < 20, f"the cached start points load fast ({cached['build_s']:.1f} s vs {fresh['build_s']:.0f} s)")
    check(cached['digest'] == fresh['digest'], f"same start points apart from snapshot bytes (digest {fresh['digest'][:12]})")
    same = sum(a == b for a, b in zip(cached['emu'], fresh['emu']))
    check(len(cached['emu']) == len(fresh['emu']) and same == len(fresh['emu']),
          f"same emulation from {same} of {len(fresh['emu'])} snapshots (every 10th), 240 frames each")
    check(cached['play'] == fresh['play'] and cached['eps'] == fresh['eps'],
          f'same {STEPS} steps of seeded play (obs, rewards, ends, positions)')
    print('\nALL CHECKS PASS' if not bad else f'\nFAILED: {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
