#!/usr/bin/env python3
"""check_main_book_rings.py — are ring routes (a fresh start that took a giant ring, won an emerald and cleared)
recorded and verified right?

    python3 check_main_book_rings.py [RUN]      # default mg2; HCZ1's ring 5 (the agent takes it ~30%)

1. ring-share starts on HCZ1 (recording allowed from any start: test only) with the run's latest save until
   one takes the ring, wins the stage and clears: a ring candidate with per-step actions, its rings and emeralds
2. harvest: its steps replayed through the env from the same start point clear on the last step, through the
   same ring, with the same emerald: verified (kept out of the book: not from the act's start); a copy with
   its last steps cut is REJECTED
CPU only; a temp book directory. Ends with ALL CHECKS PASS or the failures.
"""
import json, os, shutil, sys, tempfile
HOME = os.path.dirname(os.path.abspath(__file__))
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    book = tempfile.mkdtemp(prefix='mgbookr_')
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_BOOK=book)
    import numpy as np, torch
    torch.set_num_threads(1)
    import coordinator as C
    import main_book as B
    from stable_baselines3 import PPO
    B.BOOK, B.INDEX = book, os.path.join(book, 'index.json')
    key = next(a for a in train_main.RUNS[run]['acts'] if a.endswith(':HCZ1'))
    env = C.SonicCoordinator(acts=[key], demo_prob=0.0)
    env.demo_info()
    env._clips = {}
    C.MG_BALANCE_TIME, C.MG_BOSS_SHARE, C.MG_FRONT_SHARE, C.MG_RING_SHARE = False, 0.0, 0.0, 1.0
    env._book_any_start = True
    model = PPO.load(train_main.paths(run)['zip'] + '.zip', device='cpu')
    print(f'1. ring-share tries on HCZ1 ({model.num_timesteps:,} steps), MG_SS={C.MG_SS}')
    cdir = os.path.join(book, 'candidates')
    got, took = None, 0
    for i in range(80):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=False)[0]))
            done = term or trunc
        took += bool(env._used_rings)
        rc = [f for f in os.listdir(cdir)] if os.path.isdir(cdir) else []
        rc = [f for f in rc if f.startswith('ring_')]
        if rc:
            got = rc[0]
            break
    check(got is not None, f'a ring candidate after {i + 1} tries ({took} took a ring): {got}')
    if not got:
        print(f'\n{len(bad)} FAILED'); sys.exit(1)
    p = os.path.join(cdir, got)
    with np.load(p) as d:
        steps, meta = d['steps'], json.loads(str(d['meta']))
    check(meta['kind'] == 'ring' and meta['emeralds'] >= 1 and meta['rings'] and len(steps) == meta['steps'],
          f"meta: rings {meta['rings']}, {meta['emeralds']} emerald(s), {meta['frames']} frames "
          f"(act {meta['act_frames']}), start point {meta['start_point']}, {len(steps)} steps")
    np.savez(os.path.join(cdir, 'ring_cut.npz'), steps=steps[:-3], meta=np.array(json.dumps(meta)))
    env.close()

    print('\n2. harvest')
    B.harvest()
    ver = sorted(os.listdir(os.path.join(cdir, 'verified'))) if os.path.isdir(os.path.join(cdir, 'verified')) else []
    rej = sorted(os.listdir(os.path.join(cdir, 'rejected'))) if os.path.isdir(os.path.join(cdir, 'rejected')) else []
    check(got in ver, f'the real one verified: {ver}')
    check(rej == ['ring_cut.npz'], f'the cut one rejected: {rej}')
    shutil.rmtree(book)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
