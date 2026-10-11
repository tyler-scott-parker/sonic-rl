#!/usr/bin/env python3
"""resplit_play_acts.py — rewrite a recorded playthrough's act list after a change to
coordinator.ActSplitter's rules (2026-10-11: Doomsday Zone never stops the act clock, so
it now clears on the first frame of the ending zone; Tyler's Doomsday recording was saved
with no clear). Unlike relabel_play_acts.py (names only), the acts' frames may change.

    python3 attic/one_off_tools/resplit_play_acts.py demos/play_<stamp>.npz          # show
    python3 attic/one_off_tools/resplit_play_acts.py demos/play_<stamp>.npz --write  # rewrite

Only the metadata changes (checked after the write: same inputs, replay check passes);
the file as recorded stays in git history."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

import demos

KEYS = ('name', 'start', 'clear', 't', 'time')


def main():
    path, write = sys.argv[1], '--write' in sys.argv[2:]
    actions, meta = demos.load_play(path)
    got = demos.replay_play(actions, meta).summary()
    old = meta['acts']
    print('recorded:', [{k: a.get(k) for k in KEYS} for a in old])
    print('replay  :', [{k: a.get(k) for k in KEYS} for a in got])
    if [{k: a.get(k) for k in KEYS} for a in got] == [{k: a.get(k) for k in KEYS} for a in old]:
        print('no change')
        return
    if not write:
        return
    meta['acts'] = got
    meta.setdefault('resplit', []).append(dict(
        at=time.strftime('%Y-%m-%d %H:%M:%S'),
        why='ActSplitter: Doomsday Zone clears on the first frame of the ending zone',
        before=[{k: a.get(k) for k in KEYS} for a in old]))
    demos.save_play(path, actions, meta)
    a2, _ = demos.load_play(path)
    assert np.array_equal(a2, actions)
    ok, note = demos.replay_play_check(path)
    print(f'written: {path} (inputs unchanged); replay check: {ok} {note}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
