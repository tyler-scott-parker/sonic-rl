#!/usr/bin/env python3
"""
chain_verify.py — replay the whole game as the main-game route book would play it, from the game's start,
and report the first segment that drifts (Tyler 2026-10-16, the full-game book: "How would this work for
'in between' areas like capsule hunts after bosses or cutscenes with no timers?").

  python3 chain_verify.py                     # his full-game recording, with the book's routes where it has one
  python3 chain_verify.py --his               # his inputs only (checks the tool: must reproduce his run)

The chain is his full-game recording's acts in order. For each act: the book's route if the book has one
for that act OF THIS RECORDING (its start state is his act start, which a chain reaches only if everything
before replayed exactly), else his own frames; between acts, the bridge: his frames from the act's clear to
the next act's start. Each segment is checked as it plays:
  act      the act starts on the frame expected and clears on the route's last frame (no death)
  bridge   the next act starts exactly where his did, relative to the bridge's start
The first failure is the drift point: from there his later inputs no longer fit the state, so that segment
is the one to remake (a clip from Tyler, agent training on it, or a script). CPU only.
"""
import argparse
import json
import os
import sys

HOME = os.path.dirname(os.path.abspath(__file__))
DEMO = 'play_20261010-202923'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--his', action='store_true', help='his inputs only (the tool check)')
    ap.add_argument('--book', default=os.path.join(HOME, 'routes_main'))
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    import numpy as np
    import coordinator as C
    import demos
    path = os.path.join(demos.DEMO_DIR, f'{DEMO}.npz')
    his, meta = demos.load_play(path)
    table = meta['action_table']
    acts = [x for x in meta['acts'] if x['clear'] is not None]
    try:
        ix = json.load(open(os.path.join(a.book, 'index.json')))
    except (OSError, ValueError):
        ix = dict(acts={})
    book = {}
    if not a.his:
        for key, e in ix['acts'].items():
            if e.get('file') and e.get('demo') == DEMO:
                with np.load(os.path.join(a.book, e['file'])) as d:
                    book[e['act']] = (d['actions'].astype(np.int64), e)
    env = C.make_retro()
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    split = C.ActSplitter()
    frame = 0                                   # frames played in this chain

    def play(buttons):
        nonlocal frame
        _, info = C.emu_frame(env, buttons)
        ev = split.feed(frame, info)
        frame += 1
        return ev, info

    # up to the first act's start: his frames
    for i in range(acts[0]['start'] + 1):
        play(table[his[i]])
    print(f"chain from {DEMO}'s start; book routes used: {sorted(book) or 'none'}")
    for n, act in enumerate(acts):
        name = act['name']
        if split.cur is None or split.cur['name'] != name:
            print(f"DRIFT at {name}: expected it to start at chain frame {frame - 1}, the game is in "
                  f"{split.cur['name'] if split.cur else 'no act'}")
            return 1
        if name in book:
            route, e = book[name]
            # the book route starts at his act's first start point (frame start_frame of his run)
            lead = e['start_frame'] - act['start']
            for i in range(act['start'] + 1, act['start'] + 1 + lead):
                play(table[his[i]])
            src, steps = f"book ({e['frames']} frames, his {e.get('his', '?')})", [C.ACTION_BUTTONS[int(x)] for x in route]
        else:
            src, steps = 'his frames', [table[his[i]] for i in range(act['start'] + 1, act['clear'] + 1)]
        ev, mine = None, name in book             # his own frames may include deaths and reloads
        for j, b in enumerate(steps):
            ev, info = play(b)
            if ev == 'clear' or (ev == 'death' and mine):
                break
        if ev != 'clear' or j != len(steps) - 1:
            print(f"DRIFT in {name} ({src}): {ev or 'no event'} at step {j + 1} of {len(steps)}")
            return 1
        print(f"  ok  {name:6s} {src}: cleared on its last frame (chain frame {frame - 1})")
        if n + 1 == len(acts):
            break
        nxt = acts[n + 1]
        bridge = [table[his[i]] for i in range(act['clear'] + 1, nxt['start'] + 1)]
        started = None
        for j, b in enumerate(bridge):
            ev, info = play(b)
            if ev == 'start':
                started = j
                break
        if started != len(bridge) - 1:
            print(f"DRIFT in the bridge {name} -> {nxt['name']} (his {len(bridge)} frames): "
                  + (f'the next act started after {started + 1}' if started is not None else 'the next act never started'))
            return 1
        print(f"  ok  bridge {name} -> {nxt['name']}: {len(bridge)} frames, next act on time")
    print(f'ALL {len(acts)} ACTS AND BRIDGES REPLAY ({frame:,} frames)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
