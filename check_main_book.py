#!/usr/bin/env python3
"""check_main_book.py — does the passive main-game route book (MG_BOOK, main_book.py) keep exactly the right
episodes and promote only routes that replay?

    python3 check_main_book.py [RUN]          # default mg2; a temp book directory, the run's latest save

1. recording: with MG_BOOK set, only fresh starts (an act's first start point, drawn as such) record their
   frames; boss/ring/front-share and clip starts, and forced starts (a tool's key/ckpt), don't; with MG_BOOK
   empty nothing is recorded
2. a real candidate: fresh AIZ1 tries of the run's save (book best set out of reach) until one clears; its
   file has one action per frame of the episode
3. harvest: it replays to the act's clear on exactly its last frame and is PROMOTED (index + file); copies
   with frames cut off ("never cleared") or added ("cleared ... early") are REJECTED; a candidate slower than
   the book is kept out
CPU only. Ends with ALL CHECKS PASS or the failures.
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
    book = tempfile.mkdtemp(prefix='mgbook_')
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_BOOK=book)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    import main_book as B
    from stable_baselines3 import PPO
    B.BOOK, B.INDEX = book, os.path.join(book, 'index.json')
    aiz1 = next(a for a in train_main.RUNS[run]['acts'] if a.endswith(':AIZ1'))
    env = C.SonicCoordinator(acts=[aiz1], demo_prob=0.0)
    env.demo_info()
    seg = env._segs[0]
    key = seg['key']

    print('1. what records')
    env.set_demo_pointers({key: 0})
    C.MG_BOSS_SHARE = C.MG_RING_SHARE = C.MG_FRONT_SHARE = 0.0
    env.reset()
    check(env._book_frames == [] and env._k == 0, f'a fresh start records (start point {env._k})')
    env.reset(options={'key': key, 'ckpt': 0})
    check(env._book_frames is None, 'a forced start (key/ckpt) does not')
    C.MG_RING_SHARE = 1.0
    env.reset()
    check(env._book_frames is None, f'a ring-share start does not (start {env._k}, clip {env._clip})')
    C.MG_RING_SHARE = 0.0
    saved = C.MG_BOOK
    C.MG_BOOK = ''
    env.reset()
    check(env._book_frames is None, 'MG_BOOK empty: nothing')
    C.MG_BOOK = saved

    print('\n2. a real candidate')
    json.dump(dict(acts={key: dict(frames=10 ** 9)}, bridges=[]), open(os.path.join(book, 'index.json'), 'w'))
    env._book_read = 0.0
    model = PPO.load(train_main.paths(run)['zip'] + '.zip', device='cpu')
    got = None
    for i in range(40):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=False)[0]))
            done = term or trunc
        res = info['mg_result']
        if res['outcome'] == 'clear' and env._book_frames is not None:
            got = (res['frames'], len(env._book_frames))
            break
    cands = sorted(os.listdir(os.path.join(book, 'candidates'))) if got else []
    check(got is not None and len(cands) == 1, f'a clear after {i + 1} tries: {got}, candidates {cands}')
    if not got:
        print('\n' + str(len(bad)) + ' FAILED'); sys.exit(1)
    p = os.path.join(book, 'candidates', cands[0])
    with np.load(p) as d:
        acts, meta = d['actions'], json.loads(str(d['meta']))
    check(len(acts) == got[0] == meta['frames'], f'{len(acts)} actions = the episode\'s {got[0]} frames')

    print('\n3. harvest')
    for tag, cut in (('short', acts[:-30]), ('long', np.concatenate([acts, acts[-20:]]))):
        np.savez(os.path.join(book, 'candidates', f'AIZ1_{tag}.npz'), actions=cut,
                 meta=np.array(json.dumps(dict(meta, frames=len(cut)))))
    slow = dict(meta)
    np.savez(os.path.join(book, 'candidates', 'AIZ1_zslow.npz'), actions=acts, meta=np.array(json.dumps(slow)))
    env.close()
    B.harvest()
    ix = json.load(open(os.path.join(book, 'index.json')))
    e = ix['acts'][key]
    check(e['frames'] == len(acts) and e.get('file') and os.path.exists(os.path.join(book, e['file'])),
          f"promoted: {e['frames']} frames, {e.get('file')}")
    rej = sorted(os.listdir(os.path.join(book, 'candidates', 'rejected')))
    ver = sorted(os.listdir(os.path.join(book, 'candidates', 'verified')))
    check(rej == ['AIZ1_long.npz', 'AIZ1_short.npz'], f'rejected: {rej}')
    check('AIZ1_zslow.npz' in ver and len(ver) == 2, f'verified (promoted + kept out as not faster): {ver}')
    shutil.rmtree(book)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
