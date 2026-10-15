#!/usr/bin/env python3
"""build_bc_dataset.py — Tyler's play as (observation, action) pairs, to pretrain the main-game agent
to copy him before reinforcement learning (Tyler 2026-10-12: "This sounds great!" to imitation first).

    python3 build_bc_dataset.py [OUT_DIR] [--demos play_20260930-153946,play_20261010-202923,play_20261011-095800]

Replays each recording frame by frame through the env's own emu_frame (CPU) and keeps, per emulator
frame of the recording, the 84x84 observation frame process_frame makes (the max of that frame and the
one before, grayscale), so any frame's stack [p(i-12), p(i-8), p(i-4), p(i)] is exactly what the agent
sees on a decision right after frame i. A sample is every frame i inside a cleared act (in the level,
not dying): the label is the agent action (of the 9, each held 4 frames) that agrees with his buttons on
the most of frames i+1..i+4, so every alignment of his presses to the agent's 4-frame grid is covered
(quantized_his.py: his inputs rounded this way clear AIZ1's climb). Also kept: how many of the 4 frames
agree, the act, its demo time t and length T (the value head's target comes from T - t).

Output (not in git): OUT_DIR/frames_<demo>.u8 (n_frames x 84 x 84, memmap), OUT_DIR/samples.npz
(demo index, frame, action, agree, act index, t, T, timer_on) and OUT_DIR/meta.json.
CPU only, one core: ~5 min and ~3 GB for the whole game.
"""
import argparse
import json
import os
import sys
import time

HOME = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMOS = 'play_20260930-153946,play_20261010-202923,play_20261011-095800'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?', default=os.path.join(HOME, 'datasets', 'bc_v1'))
    ap.add_argument('--demos', default=DEFAULT_DEMOS)
    ap.add_argument('--air-bar', action='store_true', help="draw Sonic's air into the frames (MG_AIR_BAR)")
    ap.add_argument('--compass', action='store_true', help='draw the ring compass into the frames (MG_RING_COMPASS)')
    ap.add_argument('--boss-bar', action='store_true', help="draw the boss's hits left into the frames (MG_BOSS_BAR)")
    a = ap.parse_args()
    os.environ['MG_AIR_BAR'] = '1' if a.air_bar else '0'
    os.environ['MG_RING_COMPASS'] = '1' if a.compass else '0'
    os.environ['MG_BOSS_BAR'] = '1' if a.boss_bar else '0'
    os.environ.setdefault('MG_ENV_VERSION', '2')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MG_LOG'] = os.devnull
    sys.path.insert(0, HOME)
    import numpy as np
    import coordinator as C
    import demos as demo_io

    B = [0, 6, 7, 5, 4]                                     # B (jump), LEFT, RIGHT, DOWN, UP
    agent = np.array([[m[b] for b in B] for m in C.ACTION_BUTTONS], dtype=bool)   # 9 x 5
    os.makedirs(a.out, exist_ok=True)
    base = C.make_retro()                                   # an emulator to replay with
    cols = {k: [] for k in ('demo', 'frame', 'action', 'agree', 'act', 't', 'T', 'timer_on')}
    acts_names, meta = [], {'demos': [], 'air_bar': C.MG_AIR_BAR, 'ring_compass': C.MG_RING_COMPASS,
                            'boss_bar': C.MG_BOSS_BAR,
                            'actions': C.ACTION_NAMES, 'frame_skip': C.FRAME_SKIP,
                            'stack': C.STACK, 'obs_size': C.OBS_SIZE}
    t0 = time.time()
    for di, did in enumerate(a.demos.split(',')):
        path = os.path.join(HOME, 'demos', f'{did}.npz')
        actions, m = demo_io.load_play(path)
        table = np.asarray(m['action_table'], dtype=bool)
        his = table[np.asarray(actions)][:, B]              # frames x 5
        n = len(actions)
        frames = np.lib.format.open_memmap(os.path.join(a.out, f'frames_{did}.npy'), mode='w+',
                                           dtype=np.uint8, shape=(n, C.OBS_SIZE, C.OBS_SIZE))
        base.reset()
        base.em.set_state(demo_io.play_start_state(m))
        split, prev, rows, cur = C.ActSplitter(), None, [], None
        bt = C.BossTracker() if C.MG_BOSS_BAR else None   # the boss bar as the env draws it
        for i, act in enumerate(actions):
            raw, info = C.emu_frame(base, table[act])
            if bt is not None:
                bt.update(base, info, 1)
            frames[i] = C.obs_frame(prev, raw, info, boss=bt.bar() if bt is not None else None)
            prev = raw
            event = split.feed(i, info)
            if event == 'start':
                cur = split.cur['name']
            elif event == 'clear':
                done = split.acts[-1]
                rows = [r + (done['t'],) if r[4] == done['name'] else r for r in rows]
                cur = None
                continue
            if cur is None or i + C.FRAME_SKIP >= n:
                continue
            in_level, _, dead, timer_on = C.frame_flags(info)
            if not in_level or dead:
                continue
            win = his[i + 1:i + 1 + C.FRAME_SKIP]
            agree = (agent[:, None, :] == win[None, :, :]).all(-1).sum(-1)
            k = int(np.argmax(agree))
            rows.append((di, i, k, int(agree[k]), cur, int(split.cur['t']), bool(timer_on)))
        frames.flush()
        kept = [r for r in rows if len(r) == 8]             # in an act that was cleared
        for r in kept:
            if r[4] not in acts_names:
                acts_names.append(r[4])
            for key, v in zip(('demo', 'frame', 'action', 'agree', 'act', 't', 'timer_on', 'T'), r):
                cols[key].append(acts_names.index(v) if key == 'act' else v)
        meta['demos'].append(dict(id=did, frames=n, samples=len(kept),
                                  acts=sorted({r[4] for r in kept}, key=[x[4] for x in kept].index)))
        print(f'{did}: {n:,} frames, {len(kept):,} samples in {len({r[4] for r in kept})} acts '
              f'[{time.time() - t0:.0f} s]', flush=True)
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    np.savez(os.path.join(a.out, 'samples.npz'), **arrs)
    meta['acts'] = acts_names
    counts = np.bincount(arrs['action'], minlength=len(C.ACTION_NAMES))
    meta['action_counts'] = {C.ACTION_NAMES[i]: int(c) for i, c in enumerate(counts)}
    meta['agree_counts'] = {int(v): int((arrs['agree'] == v).sum()) for v in range(5)}
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(f"{len(arrs['action']):,} samples; actions {meta['action_counts']}; frames of 4 agreeing with the "
          f"label {meta['agree_counts']} [{time.time() - t0:.0f} s]")


if __name__ == '__main__':
    main()
