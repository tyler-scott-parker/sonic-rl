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

Ring routes: a fresh start that entered a giant ring, won an emerald and cleared is saved as per-step agent
actions (the special-stage pilot plays the stage itself, deterministically, from the special-stage route book);
`harvest` replays the steps through the training env from the same start: it must clear on the last step,
in the same frames, through the same rings, with the same emeralds. The fastest (real frames, stage
included) per set of rings goes in the index's 'rings'.

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
        return dict(acts={}, bridges=[], rings={})


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


def verify_steps(C, env, seg, meta, steps):
    """Replay a ring route's per-step actions through the env from its start point: (ok, why)."""
    env.reset(options={'key': seg['key'], 'ckpt': meta['start_point']})
    info, term, trunc, j = {}, False, False, -1
    for j, a in enumerate(steps):
        _, _, term, trunc, info = env.step(int(a))
        if term or trunc:
            break
    res = info.get('mg_result') or {}
    rings = sorted(list(map(int, r)) for r in env._used_rings)
    em = sum(st.get('emeralds', 0) for st in env._stages)
    if res.get('outcome') != 'clear':
        return False, f"{res.get('outcome') or 'no end'} at step {j + 1} of {len(steps)}"
    if j != len(steps) - 1:
        return False, f'cleared at step {j + 1}, {len(steps) - 1 - j} early'
    if env.frames != meta['frames'] or rings != meta['rings'] or em != meta['emeralds']:
        return False, (f"cleared, but {env.frames} frames / rings {rings} / {em} emeralds vs recorded "
                       f"{meta['frames']} / {meta['rings']} / {meta['emeralds']}")
    return True, f'clears on its last step through rings {rings}, {em} emerald(s), {env.frames} frames'


def harvest_ring(C, env, segs, ix, p, steps, meta):
    seg = segs.get(meta['key'])
    if seg is None:
        ok, why = False, f"act {meta['key']} not in {RUN}"
    else:
        ok, why = verify_steps(C, env, seg, meta, steps)
    ix.setdefault('rings', {})
    best = ix['rings'].get(meta['ring_key'], {}).get('frames', float('inf'))
    if ok and meta['start_point'] == 0 and meta['frames'] < best:
        dest = os.path.join(BOOK, f"ring_{meta['act']}_{meta['ring_key'].replace(';', '_').replace(',', '-')}_{meta['frames']}.npz")
        shutil.copy2(p, dest)
        ix['rings'][meta['ring_key']] = dict(act=meta['act'], key=meta['key'], demo=meta['demo'], rings=meta['rings'],
                                             frames=meta['frames'], act_frames=meta['act_frames'],
                                             emeralds=meta['emeralds'], file=os.path.basename(dest),
                                             source=os.path.basename(p), added=time.strftime('%Y-%m-%d %H:%M:%S'))
        save_index(ix)
        print(f"PROMOTED ring route {meta['act']} {meta['ring_key']}: {meta['frames']} frames "
              f"(act time {meta['act_frames']}, his act {meta['his_frames']}): {why}")
        return 'verified'
    if ok:
        print(f"kept out {os.path.basename(p)}: {why}, but "
              + ('not from the act start (start point %d)' % meta['start_point'] if meta['start_point'] != 0
                 else f'not faster than the book ({best})'))
        return 'verified'
    print(f'REJECTED {os.path.basename(p)}: {why}')
    return 'rejected'


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
            meta = json.loads(str(d['meta']))
            actions = d['actions'] if 'actions' in d.files else None
            steps = d['steps'] if 'steps' in d.files else None
        if meta.get('kind') == 'ring':
            where = harvest_ring(C, env, segs, ix, p, steps, meta)
            os.makedirs(os.path.join(BOOK, 'candidates', where), exist_ok=True)
            shutil.move(p, os.path.join(BOOK, 'candidates', where, os.path.basename(p)))
            continue
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
    for k, v in ix.get('rings', {}).items():
        print(f"ring route {v['act']} through {k}: {v['frames']} frames (act time {v['act_frames']}), "
              f"{v['emeralds']} emerald(s)  {v['source']}")
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
