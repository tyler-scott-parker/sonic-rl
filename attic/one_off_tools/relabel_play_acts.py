#!/usr/bin/env python3
"""relabel_play_acts.py — rename a recorded playthrough's acts after a change
to coordinator.act_name / part_name (2026-10-10: internal zones $16/$17 were
both 'HPZ', and LBZ2's timer restart after its cutscene made a second 'LBZ2').

    python3 attic/one_off_tools/relabel_play_acts.py demos/play_<stamp>.npz          # show
    python3 attic/one_off_tools/relabel_play_acts.py demos/play_<stamp>.npz --write  # rewrite

Replays the recording with today's ActSplitter and requires every act to match
the recorded one in everything but its name (start, clear, own time, in-game
time, deaths, stages, zone). Only then, with --write, replaces the names in the
file's metadata; the inputs are untouched (checked after the write). The file
as recorded stays in git history.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

import demos

FIELDS = ('start', 'clear', 't', 'time', 'deaths', 'stages', 'zone')


def main():
    path, write = sys.argv[1], '--write' in sys.argv[2:]
    actions, meta = demos.load_play(path)
    got = demos.replay_play(actions, meta).summary()
    old = meta['acts']
    if len(got) != len(old) or any(g[f] != o[f] for g, o in zip(got, old) for f in FIELDS):
        print('!! the replay splits the acts differently, not just their names: nothing written')
        for g, o in zip(got, old):
            print('  ', {f: o[f] for f in ('name',) + FIELDS}, '\n   ', {f: g[f] for f in ('name',) + FIELDS})
        sys.exit(1)
    renames = [(k, o['name'], g['name']) for k, (g, o) in enumerate(zip(got, old)) if g['name'] != o['name']]
    for k, a, b in renames:
        print(f'  act {k:2d} at frame {old[k]["start"]:7d}: {a} -> {b}')
    names = [g['name'] for g in got]
    dup = sorted({n for n in names if names.count(n) > 1})
    print(f'{len(renames)} rename(s); duplicate names after: {dup or "none"}')
    if not write or not renames:
        return
    meta['acts'] = [dict(o, name=g['name']) for g, o in zip(got, old)]
    meta.setdefault('relabelled', []).append(dict(
        at=time.strftime('%Y-%m-%d %H:%M:%S'), why='coordinator act names fixed: $16/$17 zones, cutscene timer restarts',
        renames=[[k, a, b] for k, a, b in renames]))
    demos.save_play(path, actions, meta)
    a2, m2 = demos.load_play(path)
    assert np.array_equal(a2, actions) and [a['name'] for a in m2['acts']] == names
    print(f'written: {path} (inputs unchanged, {len(a2):,} frames)')


if __name__ == '__main__':
    main()
