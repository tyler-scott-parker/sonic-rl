#!/usr/bin/env python3
"""
main_book.py — the main game's route book: for each act, the fastest verified clear from the act's start
(Tyler 2026-10-16: "Would a route book be useful for the main game?" ... "Yes that sounds great").

  python3 main_book.py                 # status: each act's best vs his time, bridges, candidates waiting
  python3 main_book.py seed            # (once) his recordings as the first entries + the bridges between acts
  python3 main_book.py harvest         # verify waiting candidates by replay; promote the faster ones

An act's entry is an input sequence (one ACTION_BUTTONS index per frame) from the act's first start point
(his saved state there) that replays to the act's clear (the in-game timer stops with Sonic alive) on its
last frame, with no death and no special stage on the way. Seeded with his own runs (no files: his
recording is the route); the training env (MG_BOOK, RUNS 'book') saves fresh clears faster than the entry
to BOOK/candidates/, and `harvest` replays each from the same state: one that clears on exactly its last
frame and beats the entry becomes BOOK/<act>.npz; the rest go to candidates/rejected/ with the reason.

Bridges are the stretches between one act's clear and the next act's start in his full-game recording
(results screens, capsule hunts, cutscenes, the walk into act 2): kept as his frames for now. A full-game run
chained from the book needs them remade from the book's own act endings where his inputs drift; that is
for later (a whole-chain verifier will say which).
CPU only; `harvest` builds the run's acts from the recordings first (a minute or two).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import zlib

HOME = os.path.dirname(os.path.abspath(__file__))
RUN = 'mg2'
BOOK = os.path.join(HOME, 'routes_main')
INDEX = os.path.join(BOOK, 'index.json')


def load_index():
    try:
        return json.load(open(INDEX))
    except (OSError, ValueError):
        return dict(acts={}, bridges=[])


def save_index(ix):
    os.makedirs(BOOK, exist_ok=True)
    with open(INDEX + '.tmp', 'w') as f:
        json.dump(ix, f, indent=1)
    os.replace(INDEX + '.tmp', INDEX)


def build_env():
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(RUN)
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_BOOK='')
    import coordinator as C
    env = C.SonicCoordinator(demo_prob=0.0)
    env.demo_info()
    return C, env


def seed():
    """His runs as each act's first entry (from the act's first start point to his clear), and every bridge
    of his recordings (act clear -> next act start, his frames)."""
    C, env = build_env()
    import demos
    ix = load_index()
    for s in env._segs:
        if s['key'] not in ix['acts']:
            ck = s['ckpts'][0]
            ix['acts'][s['key']] = dict(act=s['name'], demo=s['demo'], frames=int(s['T'] - ck['t']),
                                        his=int(s['T'] - ck['t']),
                                        source='his recording', start_frame=int(ck['frame']), file=None,
                                        added=time.strftime('%Y-%m-%d %H:%M:%S'))
    bridges = []
    for path in env.demo_files:
        meta = demos.load_play(path)[1]
        acts = meta['acts']
        for a, b in zip(acts, acts[1:]):
            if a['clear'] is None:
                continue
            bridges.append(dict(demo=demos.demo_id(path), after=a['name'], before=b['name'],
                                from_frame=int(a['clear']), to_frame=int(b['start']),
                                frames=int(b['start'] - a['clear'])))
    ix['bridges'] = bridges
    save_index(ix)
    print(f"seeded: {len(ix['acts'])} acts, {len(bridges)} bridges -> {os.path.relpath(INDEX, HOME)}")


def verify(C, env, seg, actions):
    """Replay `actions` from the act's first start point: (ok, why, frames)."""
    env.base_env.reset()
    env.base_env.em.set_state(zlib.decompress(seg['ckpts'][0]['state']))
    for i, a in enumerate(actions):
        _, info = C.emu_frame(env.base_env, C.ACTION_BUTTONS[int(a)])
        in_level, in_stage, dead, timer_on = C.frame_flags(info)
        if in_stage:
            return False, f'a special stage at frame {i}', i + 1
        if dead:
            return False, f'died at frame {i}', i + 1
        if in_level and not timer_on:
            if i == len(actions) - 1:
                return True, 'clears on its last frame', i + 1
            return False, f'cleared at frame {i}, {len(actions) - 1 - i} frames early', i + 1
    return False, 'never cleared', len(actions)


def harvest():
    cands = sorted(glob.glob(os.path.join(BOOK, 'candidates', '*.npz')))
    if not cands:
        print('no candidates waiting')
        return
    C, env = build_env()
    import numpy as np
    ix = load_index()
    segs = {s['key']: s for s in env._segs}
    for p in cands:
        with np.load(p) as d:
            actions, meta = d['actions'], json.loads(str(d['meta']))
        seg = segs.get(meta['key'])
        why, ok = None, False
        if seg is None:
            why = f"act {meta['key']} not in {RUN}"
        elif meta.get('action_table') != C.ACTION_BUTTONS:
            why = 'recorded with a different action table'
        elif int(seg['ckpts'][0]['frame']) != meta['start_frame']:
            why = f"start point moved (frame {seg['ckpts'][0]['frame']} now, {meta['start_frame']} then)"
        else:
            ok, why, n = verify(C, env, seg, actions)
        best = ix['acts'].get(meta['key'], {}).get('frames', meta['his_frames'])
        if ok and len(actions) < best:
            name = f"{meta['act']}_{len(actions)}"
            dest = os.path.join(BOOK, f'{name}.npz')
            shutil.copy2(p, dest)
            his = ix['acts'].get(meta['key'], {}).get('his', meta['his_frames'])
            ix['acts'][meta['key']] = dict(act=meta['act'], demo=meta['demo'], frames=int(len(actions)), his=int(his),
                                           source=os.path.basename(p), start_frame=meta['start_frame'],
                                           file=os.path.basename(dest), added=time.strftime('%Y-%m-%d %H:%M:%S'),
                                           beat=int(best))
            save_index(ix)
            where = 'verified'
            print(f"PROMOTED {meta['act']}: {len(actions)} frames (was {best}, his {meta['his_frames']}): {why}")
        else:
            where = 'verified' if ok else 'rejected'
            print(f"{'kept out' if ok else 'REJECTED'} {os.path.basename(p)}: {why}"
                  + (f' but not faster than the book ({best})' if ok else ''))
        os.makedirs(os.path.join(BOOK, 'candidates', where), exist_ok=True)
        shutil.move(p, os.path.join(BOOK, 'candidates', where, os.path.basename(p)))


def status():
    ix = load_index()
    if not ix['acts']:
        print('empty book: run  python3 main_book.py seed')
        return
    print(f"{'act':6s} {'book':>6s} {'his':>6s} {'gain':>6s}  source")
    for k, v in ix['acts'].items():
        his = v.get('his', v['frames'])
        print(f"{v['act']:6s} {v['frames']:6d} {his:6d} {his - v['frames']:+6d}  {v['source']}")
    n = len(glob.glob(os.path.join(BOOK, 'candidates', '*.npz')))
    print(f"bridges: {len(ix['bridges'])} (his frames); candidates waiting: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', nargs='?', default='status', choices=['status', 'seed', 'harvest'])
    a = ap.parse_args()
    os.chdir(HOME)
    {'status': status, 'seed': seed, 'harvest': harvest}[a.cmd]()


if __name__ == '__main__':
    main()
