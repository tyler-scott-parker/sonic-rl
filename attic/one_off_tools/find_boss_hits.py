#!/usr/bin/env python3
"""find_boss_hits.py — where does S3K keep a boss's hits-left count? Found
from Tyler's full-game recording rather than assumed (Tyler 2026-10-14: "go
ahead and find the boss hit counter").

Replays the recording through coordinator.emu_frame and, around every boss
fight, keeps the object RAM (110 slots of $4A bytes from $B000) frame by
frame. A fight window = Boss_flag up (gaps under FIGHT_GAP merged) plus
AFTER frames once it drops; the acts the game never raises Boss_flag for
(MGZ1, SOZ1, SOZ2, SSZ1, DEZ3) get their last NOFLAG_LOOK frames before the
clear instead. In each window every (slot, byte offset) is scored as a
hits-left counter: it only ever steps down by exactly 1 (one spawn-time jump
up from 0 allowed), MIN_HITS to MAX_HITS times, never two steps within
MIN_GAP frames (a hit boss flashes, invulnerable; a timer ticks far more
often). The report gives, per fight, the
best candidates with when they stepped down (relative to the act's clear,
which for an act-ending boss is its last hit), and what the slot looked like
(code pointer, collision_flags at +$28).

    python3 attic/one_off_tools/find_boss_hits.py [--acts AIZ1,AIZ2,...] [--out logs/boss_hits_scan.json]
CPU only; ~3-4 min for the whole game.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np                                            # noqa: E402

import coordinator as C                                       # noqa: E402
import demos                                                  # noqa: E402

DEMO = 'demos/play_20261010-202923.npz'
SLOTS, SLOT_LEN, OBJ0 = 110, 0x4A, 0xB000
AFTER, NOFLAG_LOOK = 240, 3600       # frames kept after Boss_flag drops; looked back from a no-flag act's clear
NOFLAG = {'MGZ1', 'SOZ1', 'SOZ2', 'SSZ1', 'DEZ3'}
MIN_HITS, MAX_HITS, MIN_GAP = 3, 16, 8

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--acts', help='only these acts (comma-separated names)')
ap.add_argument('--out', default='logs/boss_hits_scan.json')
args = ap.parse_args()
only = set(args.acts.split(',')) if args.acts else None


def oriented(env, info, cache={}):
    """The 64 KB of work RAM in 68k byte order (the core may hand it over
    byte-swapped; decided once, by finding Player_1's x/y, as coordinator does)."""
    raw = np.frombuffer(bytes(env.get_ram()), dtype=np.uint8)
    if 'swap' not in cache:
        cache['swap'] = not np.array_equal(C._ram_words(env, info), raw)
        print(f"RAM byte order: {'swapped' if cache['swap'] else 'as is'}", flush=True)
    return raw.reshape(-1, 2)[:, ::-1].reshape(-1) if cache['swap'] else raw


def candidates(block):
    """block: (frames, SLOTS, SLOT_LEN) uint8. Every (slot, offset) that acts
    like a hits-left counter, best first."""
    d = np.diff(block.astype(np.int16), axis=0)               # (frames-1, SLOTS, SLOT_LEN)
    down1 = (d == -1).sum(0)
    up = d > 0
    other = ((d != 0) & (d != -1) & ~up).sum(0)
    ups_from0 = (up & (block[:-1] == 0)).sum(0)
    ups = up.sum(0)
    ok = ((down1 >= MIN_HITS) & (down1 <= MAX_HITS) & (other == 0)
          & (ups - ups_from0 == 0) & (ups_from0 <= 1))
    out = []
    for s, o in zip(*np.nonzero(ok)):
        v = block[:, s, o]
        steps = (np.flatnonzero(d[:, s, o] == -1) + 1).tolist()
        if min(np.diff(steps), default=MIN_GAP) < MIN_GAP:
            continue
        out.append(dict(slot=int(s), off=int(o), hits=int(down1[s, o]), start=int(v[steps[0] - 1]),
                        end=int(v[-1]), steps=steps))
    out.sort(key=lambda c: (-c['hits'], c['end'] != 0, c['off']))
    return out


def report(win, act, clear, blk, frames):
    cands = candidates(blk)
    rel = (lambda f: f - clear) if clear is not None else (lambda f: f)
    first, last = frames[0], frames[-1]
    head = (f"{act}: window {rel(first):+d}..{rel(last):+d} frames from the clear "
            f"({len(frames) / 60:.0f} s{', no Boss_flag' if win['noflag'] else ''})")
    print(head, flush=True)
    rows = []
    for c in cands[:6]:
        at = [frames[i] for i in c['steps']]
        s0 = c['steps'][0]
        slot = blk[s0, c['slot']]
        code = int.from_bytes(bytes(slot[0:4]), 'big')
        rows.append(dict(c, at=[rel(f) for f in at], code=f'{code:06X}',
                         coll=f"{int(blk[s0 - 1, c['slot'], 0x28]):02X}"))
        print(f"   slot {c['slot']:3d} +${c['off']:02X}: {c['start']} -> {c['end']} in {c['hits']} steps of 1, at "
              + ' '.join(f'{rel(f):+d}' for f in at)
              + f"   (code {code:06X}, +$28 before the first: {int(blk[s0 - 1, c['slot'], 0x28]):02X})", flush=True)
    if not cands:
        print('   no candidate', flush=True)
    return dict(act=act, first=rel(first), last=rel(last), noflag=win['noflag'], candidates=rows)


def main():
    actions, meta = demos.load_play(DEMO)
    table = meta['action_table']
    env = C.make_retro()
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    acts = [(a['name'], a['start'], a['clear']) for a in meta['acts'] if a['clear'] is not None]

    def act_at(i):
        for name, s, c in acts:
            if s <= i <= c + AFTER:
                return name, c
        return None, None

    noflag_windows = {name: (c - NOFLAG_LOOK, c + 60) for name, s, c in acts if name in NOFLAG}
    results, win, buf = [], None, []

    def close():
        nonlocal win, buf
        results.append(report(win, win['act'], win['clear'], np.stack(buf), win['frames']))
        win, buf = None, []

    t0 = time.time()
    for i, a in enumerate(actions):
        _, info = C.emu_frame(env, table[a])
        act, clear = act_at(i)
        if only is not None and act not in only:
            act = None
        nf = act in noflag_windows and noflag_windows[act][0] <= i <= noflag_windows[act][1]
        on = act is not None and (nf or int(info['mg_boss']) != 0)
        if win is not None and (win['act'] != act and on):
            close()                                     # another act's fight
        if on:
            if win is None:
                win = dict(act=act, clear=clear, frames=[], noflag=nf)
            win['last_on'] = i
        if win is not None:
            if i - win['last_on'] > (0 if win['noflag'] else AFTER):
                close()
            else:
                ram = oriented(env, info)
                buf.append(ram[OBJ0:OBJ0 + SLOTS * SLOT_LEN].reshape(SLOTS, SLOT_LEN).copy())
                win['frames'].append(i)
        if i % 50000 == 0:
            print(f'[frame {i:,} of {len(actions):,}, {time.time() - t0:.0f} s]', flush=True)
    if win is not None:
        close()
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
