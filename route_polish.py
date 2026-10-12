#!/usr/bin/env python3
"""route_polish.py — frame-exact search for a faster finish of a book route
(CPU, beside training; no model).

    python3 route_polish.py BOARD FROM_FRAME [--beam 3000] [--per 8] [--out candidates]
    python3 route_polish.py 9 2700

From frame FROM_FRAME of the board's route-book route, breadth-first over EVERY
input (the 7 actions) on every frame, so it finds tricks no agent was trained
to try. The stage is deterministic, so a path found here is a route. Game
states that are identical are merged: all RAM except the input-history ring at
$E500-$E5FF and the controller bytes $F604/$F605 (a button press leaves a copy
there for 64 frames while play itself is unchanged), but the jump button's
held bit is kept, since it decides whether the next press is a press. A
direction pressed mid-cell IS kept: it buffers a turn ($E42A). Above `beam`
states per frame, the ones with fewer spheres left and nearer a blue sphere are
kept, at most `per` per (cell, heading, jump state) first so that one idea
can't fill the beam. Each kept state costs ~1 MB of RAM.

It searches up to the book's own time: finding exactly the book's time means
the beam kept the book's path and nothing within it is faster (the sanity
check). A faster clear (the book's frames up to FROM_FRAME + the found frames)
is saved to routes/candidates/, where `route_book.py harvest` checks and
promotes it like any other.

Cost: ~26 s per frame at beam 3000 (one core), so a 100-frame finish ~45 min.
Measured 2026-10-12: from 3,830 on board 3 and 2,880 on board 9 it finds
exactly Tyler's 3,944 and 2,988 (nothing faster in those finishes).
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
INPUT_BYTES = list(range(0xE500, 0xE600)) + [0xF604, 0xF605]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board', type=int)
    ap.add_argument('start', type=int, help='frame of the book route to search from')
    ap.add_argument('--beam', type=int, default=3000)
    ap.add_argument('--per', type=int, default=8)
    ap.add_argument('--out', default=None, help='directory for a faster route (default routes/candidates)')
    a = ap.parse_args()
    sys.path.insert(0, HOME)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import demos
    import route_book
    import ss_pilot
    sse = ss_pilot.import_env(ss_pilot.run_preset('d10'))
    B, F0, K = a.board, a.start, a.beam
    env = sse.SpecialStageEnv()
    env._replaying = True
    book, meta = demos.load_demo(route_book.book_path(B))
    env.reset(options={'stage_file': sse.stage_file(B), 'replay': True})
    for act in book[:F0]:
        env._step_frames(int(act), 1)
    em, base = env.base_env.em, env.base_env
    buttons = [env._actions[i] for i in range(len(sse.ACTION_BUTTONS))]

    def key_of(ram):
        k = ram.copy()
        b_held = k[0xF604] & 0x10
        k[INPUT_BYTES] = 0
        return hashlib.sha1(k.tobytes() + bytes([b_held])).digest()

    def dist_of(ram, cell):
        """Torus Manhattan distance from Sonic's cell to the nearest blue sphere."""
        grid = ram[sse.GRID_BASE: sse.GRID_BASE + 0x400].reshape(32, 32)
        blues = np.argwhere(grid == 2)
        if not len(blues):
            return 0
        dr, dc = np.abs(blues[:, 0] - cell[0]), np.abs(blues[:, 1] - cell[1])
        return int((np.minimum(dr, 32 - dr) + np.minimum(dc, 32 - dc)).min())

    s0 = em.get_state()
    frontier = [(s0, None)]                   # (emulator state, path node); node = (parent, action)
    t0 = time.time()
    print(f'board {B}: book {len(book)} frames; searching from frame {F0}, beam {K}, '
          f'{len(s0) // 1024} KB per state', flush=True)
    found = None
    for t in range(F0, len(book)):
        seen, new = set(), []
        for st, node in frontier:
            for i, btn in enumerate(buttons):
                em.set_state(st)
                _, _, _, _, info = base.step(btn)
                if int(info.get('blue_spheres_remaining', 999)) == 0:
                    found = ((node, i), t + 1)
                    break
                ram = env._ram()
                if ram[sse.ADDR_EXIT_TIMER]:
                    continue                    # touched a red: the stage is ending
                k = key_of(ram)
                if k in seen:
                    continue
                seen.add(k)
                cell = env._cell(int(info.get('ss_x', 0)), int(info.get('ss_y', 0)))
                new.append((em.get_state(), (node, i),
                            (int(info.get('blue_spheres_remaining', 999)), dist_of(ram, cell)),
                            (cell, int(ram[sse.ADDR_HEADING]), int(ram[sse.ADDR_JUMPING]))))
            if found:
                break
        if found:
            break
        n_distinct = len(new)
        new.sort(key=lambda x: x[2])
        if len(new) > K:
            per, keep, rest = {}, [], []
            for x in new:
                n = per.get(x[3], 0)
                (keep if n < a.per else rest).append(x)
                per[x[3]] = n + 1
            new = sorted((keep + rest)[:K], key=lambda x: x[2])
        frontier = [(x[0], x[1]) for x in new]
        if (t - F0) % 10 == 0:
            print(f'  frame {t + 1}: {n_distinct} distinct states, kept {len(frontier)}, best (left, distance) '
                  f'{new[0][2] if new else "-"} [{time.time() - t0:.0f} s]', flush=True)
    if not found:
        print(f'no clear by frame {len(book)} within the beam (the book\'s path was dropped) '
              f'[{time.time() - t0:.0f} s]')
        return
    node, frames = found
    tail = []
    while node is not None:
        node, i = node
        tail.append(i)
    route = list(book[:F0]) + tail[::-1]
    if frames == len(book):
        print(f'clear in {frames} frames = the book: nothing faster from frame {F0} '
              f'[{time.time() - t0:.0f} s]', flush=True)
        return
    print(f'FASTER: clear in {frames} frames (book {len(book)}, {len(book) - frames} faster) '
          f'[{time.time() - t0:.0f} s]', flush=True)
    m = {k: v for k, v in meta.items() if k not in ('verified', 'promoted', 'relabelled', 'format')}
    m.update(frames=len(route), recorded=time.strftime('%Y-%m-%d %H:%M'), source_tag='route_polish',
             source=f'route_polish (every input from frame {F0}, beam {K}) of board{B:02d} [{meta.get("source")}]')
    out = a.out or route_book.CAND_DIR
    path = os.path.join(out, f'b{B:02d}_{len(route)}_{os.getpid()}_{int(time.time())}.npz')
    route_book.save_atomic(path, route, m)
    print('  saved', path, '(route_book.py harvest checks and promotes it)')


if __name__ == '__main__':
    main()
