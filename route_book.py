#!/usr/bin/env python3
"""
route_book.py — a stored winning route per special-stage board, so that every
special stage the main game plays is a guaranteed emerald (Tyler 2026-10-11:
"100% fresh start completion so that the main game agent is guaranteed an
emerald each time").

A ROUTE is a demo (demos.py): one action per frame from the board's captured
state, that clears the board. The game is deterministic from the handover
(ss_pilot.py matches the speed-up timer to the captured state), so a route that
clears from the captured state clears from a giant ring every time. A sampled
policy can't promise that (100 clears in 100 tries still allows a ~3% failure
rate); a replayed route can. Measured 2026-10-11: Tyler's 14 recordings replay
frame for frame from the main game on all 14 boards (main_fixture below).

THE BOOK: routes/board{NN}.npz, the fastest verified route per board, and
routes/history/ for the ones it replaced. It starts as Tyler's recordings.
Training finds faster ones: special_stage_env saves every clear faster than
the book's route for that board to routes/candidates/ (fresh starts as they
are; demo starts with Tyler's frames up to the start point in front, so the
agent beating his last stretch is a faster whole route too). `harvest` checks
the fastest candidates and promotes them. A route enters the book only after
replaying to a clear on exactly its last frame both from the captured state
and from the main game.

    python3 route_book.py                 # the book: frames per board vs his
    python3 route_book.py seed            # start the book from Tyler's recordings
    python3 route_book.py harvest         # check candidates faster than the book; promote
    python3 route_book.py harvest --every 30   # ...again every 30 minutes (beside a training run)
    python3 route_book.py verify          # replay every route (captured + main game)

The main game picks boards 1-7 in order: the next chaos board is $FE16
(Current_special_stage) + 1 (measured: 0-6 load boards 1-7). main_fixture
builds each from AIZ1's ring with $FE16 and the emeralds of the boards before
it set; boards 8-14 come from MHZ1's ring with all 7 Chaos Emeralds (Hidden
Palace, ss_pilot.shrine). CPU only; one emulator per process, so each check
runs in a child process.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HOME = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.join(HOME, 'routes')
CAND_DIR = os.path.join(BOOK_DIR, 'candidates')
HIST_DIR = os.path.join(BOOK_DIR, 'history')
FIX = os.path.join(HOME, 'states', 'handoff')
BOARDS = range(1, 15)
HARVEST_TRIES = 5           # per board: the fastest candidates checked before giving up


def book_path(board):
    return os.path.join(BOOK_DIR, f'board{board:02d}.npz')


def load(boards=BOARDS):
    """{board: dict(actions, frames, source, meta)} of the book's routes."""
    import demos
    out = {}
    for b in boards:
        p = book_path(b)
        if os.path.exists(p):
            actions, meta = demos.load_demo(p)
            out[b] = dict(actions=actions, frames=len(actions), source=meta.get('source', '?'), meta=meta)
    return out


def book_frames():
    """{board: frames} — cheap (the trainer's envs read it for their threshold)."""
    out = {}
    for b in BOARDS:
        try:
            with np.load(book_path(b)) as d:
                out[b] = int(d['actions'].shape[0])
        except (OSError, KeyError, ValueError):
            pass
    return out


def his_recordings():
    """{board: path} of Tyler's special-stage recordings (demos/stage{N}_*.npz)."""
    out = {}
    for b in BOARDS:
        got = sorted(glob.glob(os.path.join(HOME, 'demos', f'stage{b}_*.npz')))
        if got:
            out[b] = got[-1]
    return out


def save_atomic(path, actions, meta):
    """demos.save_demo, but a reader never sees half a file (np.savez needs
    the .npz suffix on the temporary name too)."""
    import demos
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), f'.{os.getpid()}.{os.path.basename(path)}')
    demos.save_demo(tmp, actions, meta)
    os.replace(tmp, path)


# ── checks (each in a child process: one emulator per process) ───────────────

def _child(mode, args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MG_LOG=os.devnull)
    out = subprocess.run([sys.executable, __file__, '--child', mode, json.dumps(args)],
                         env=env, capture_output=True, text=True, cwd=HOME)
    lines = [l for l in out.stdout.splitlines() if l.startswith('{')]
    if out.returncode or not lines:
        raise RuntimeError(f'{mode} check crashed:\n{out.stdout[-1500:]}\n{out.stderr[-2500:]}')
    return json.loads(lines[-1])


def check_captured(paths, run='d9'):
    """{path: dict(ok, board, frames, cleared_at, spheres, total)}: each replayed
    frame by frame from its board's captured state, the game's rules (no stall
    rule). ok = the board is cleared on exactly the route's last frame."""
    return _child('captured', dict(paths=paths, run=run))


def check_main(paths, run='d9'):
    """Same, from the main game (main_fixture): ok also needs the stage to be
    the route's board, entered from a giant ring."""
    return _child('main', dict(paths=paths, run=run))


def _run_captured(paths, run):
    import demos
    import ss_pilot
    sse = ss_pilot.import_env(ss_pilot.run_preset(run))
    env = sse.SpecialStageEnv()
    env._replaying = True                   # the game's rules; no training log
    res = {}
    for p in paths:
        actions, meta = demos.load_demo(p)
        b = int(meta['stage'])
        try:
            demos.check_compatible(p, meta, sse)
        except ValueError as e:
            res[p] = dict(ok=False, board=b, frames=len(actions), why=str(e))
            continue
        env.reset(options={'stage_file': sse.stage_file(b), 'replay': True})
        ended = None
        for i, a in enumerate(actions):
            _, _, done, trunc, _ = env._step_frames(int(a), 1)
            if done or trunc:
                ended = i
                break
        cleared = bool(env._board_cleared)
        res[p] = dict(ok=cleared and ended == len(actions) - 1, board=b, frames=len(actions),
                      ended_at=ended, cleared=cleared, spheres=int(env._spheres_collected),
                      total=int(env._stage_total))
    env.close()
    print(json.dumps(res))


def main_fixture(emu, board, pilot, C):
    """Put the main game one frame before a giant ring that starts `board`,
    step the ring's frame, and (8-14) touch its Super Emerald in Hidden Palace.
    Afterwards pilot.play() is due. Boards 2-7: AIZ1's ring with $FE16 =
    board - 1 and the Chaos Emeralds of boards 1..board-1 held (count $FFB0,
    array $FFB2+ = 1), as if the earlier stages had been won."""
    for name, addr in [('ss_next', 0xFFFE16), ('em_count', 0xFFFFB0)] + [
            (f'em_arr{i}', 0xFFFFB2 + i) for i in range(7)]:
        emu.data.set_variable(name, {'address': addr, 'type': '|u1'})
    emu.reset()
    if board <= 7:
        emu.em.set_state(open(os.path.join(FIX, 'aiz1_ring_minus1.state'), 'rb').read())
        emu.data.set_value('ss_next', board - 1)
        emu.data.set_value('em_count', board - 1)
        for i in range(board - 1):
            emu.data.set_value(f'em_arr{i}', 1)
        C.emu_frame(emu, C.ACTION_BUTTONS[0])
        return None
    emu.em.set_state(open(os.path.join(FIX, 'mhz1_ring7_minus1.state'), 'rb').read())
    for i in range(7):
        emu.data.set_value(f'em_arr{i}', 2)                 # gray Super Emeralds on the pedestals
    C.emu_frame(emu, C.ACTION_BUTTONS[0])
    return pilot.shrine([board], dict(zip(C.ACTION_NAMES, C.ACTION_BUTTONS)))


def _run_main(paths, run):
    import torch
    torch.set_num_threads(1)
    import coordinator as C
    import demos
    import ss_pilot
    emu = C.make_retro()
    pilot = ss_pilot.SSPilot(emu, run=run, model=False, routes=False)
    res = {}
    for p in paths:
        actions, meta = demos.load_demo(p)
        b = int(meta['stage'])
        try:
            main_fixture(emu, b, pilot, C)
            r = pilot.play(script=actions)
        except Exception as e:
            res[p] = dict(ok=False, board=b, frames=len(actions), why=repr(e))
            continue
        res[p] = dict(ok=(r['board'] == b and r['outcome'] == 'clear'
                          and pilot.env.steps == len(actions) and r['let_go'] == 0),
                      board=b, loaded=r['board'], outcome=r['outcome'], frames=len(actions),
                      steps=pilot.env.steps, spheres=r['spheres'], total=r['total'])
    print(json.dumps(res))


# ── commands ─────────────────────────────────────────────────────────────────

def promote(path, board, why):
    """Copy a checked route into the book; the route it replaces goes to history/."""
    import demos
    actions, meta = demos.load_demo(path)
    old = book_path(board)
    if os.path.exists(old):
        _, om = demos.load_demo(old)
        os.makedirs(HIST_DIR, exist_ok=True)
        shutil.copy2(old, os.path.join(HIST_DIR, f'board{board:02d}_{om["frames"]}_{om.get("source_tag", "x")}.npz'))
    meta = dict(meta, frames=len(actions), verified=why, promoted=time.strftime('%Y-%m-%d %H:%M'))
    save_atomic(old, actions, meta)


def cmd_seed(a):
    his = his_recordings()
    paths = [his[b] for b in BOARDS if b in his]
    cap = check_captured(paths, a.run)
    main = check_main(paths, a.run)
    import demos
    for b in BOARDS:
        p = his.get(b)
        if p is None:
            print(f'board {b:2d}: no recording of Tyler\'s')
            continue
        c, m = cap[p], main[p]
        if not (c['ok'] and m['ok']):
            print(f'board {b:2d}: NOT seeded — captured {c}, main game {m}')
            continue
        if os.path.exists(book_path(b)) and not a.force:
            print(f'board {b:2d}: already in the book (seed --force replaces it)')
            continue
        actions, meta = demos.load_demo(p)
        meta = dict(meta, source=f'Tyler ({os.path.basename(p)})', source_tag='tyler',
                    his_frames=len(actions))
        tmp = os.path.join(BOOK_DIR, f'.seed{b}.npz')
        save_atomic(tmp, actions, meta)
        promote(tmp, b, dict(captured=True, main_game=True))
        os.remove(tmp)
        print(f'board {b:2d}: Tyler\'s {len(actions)} frames, clears from the captured state '
              f'and from the main game')


def cmd_harvest(a):
    while True:
        print(f'── harvest {time.strftime("%Y-%m-%d %H:%M")}', flush=True)
        harvest_once(a)
        if not a.every:
            return
        sys.stdout.flush()
        time.sleep(a.every * 60)


def harvest_once(a):
    """Per board, the candidates faster than the book, fastest first: the first
    that clears from both the captured state and the main game is promoted.
    Candidates no faster than the book afterwards are removed; ones that
    failed a check move to candidates/rejected/."""
    import demos
    book = book_frames()
    cands = {}
    for p in glob.glob(os.path.join(CAND_DIR, 'b*.npz')):
        try:
            actions, meta = demos.load_demo(p)
        except Exception:
            continue
        cands.setdefault(int(meta['stage']), []).append((len(actions), p))
    if not cands:
        print('no candidates')
    rejected = os.path.join(CAND_DIR, 'rejected')
    for b in sorted(cands):
        todo = sorted(c for c in cands[b] if c[0] < book.get(b, 10 ** 9))[:HARVEST_TRIES]
        won = None
        if todo:
            cap = check_captured([p for _, p in todo], a.run)
            good = [p for _, p in todo if cap[p]['ok']]
            main = check_main(good, a.run) if good else {}
            for f, p in todo:
                if cap[p]['ok'] and main.get(p, {}).get('ok') and won is None:
                    won = (f, p)
                elif not (cap[p]['ok'] and main.get(p, {}).get('ok')):
                    os.makedirs(rejected, exist_ok=True)
                    why = cap[p] if not cap[p]['ok'] else main.get(p)
                    print(f'board {b:2d}: candidate {os.path.basename(p)} ({f} frames) failed: {why}')
                    shutil.move(p, os.path.join(rejected, os.path.basename(p)))
        if won:
            f, p = won
            _, meta = demos.load_demo(p)
            promote(p, b, dict(captured=True, main_game=True))
            print(f'board {b:2d}: {book.get(b)} -> {f} frames ({meta.get("source")})')
            book[b] = f
        for f, p in cands[b]:
            if os.path.exists(p) and f >= book.get(b, 10 ** 9):
                os.remove(p)
    cmd_status(a)


def cmd_verify(a):
    book = load()
    paths = [book_path(b) for b in book]
    cap = check_captured(paths, a.run)
    main = check_main(paths, a.run)
    bad = 0
    for b in book:
        p = book_path(b)
        ok = cap[p]['ok'] and main[p]['ok']
        bad += not ok
        print(f'board {b:2d}: {book[b]["frames"]:5d} frames, captured state '
              f'{"clear" if cap[p]["ok"] else "FAIL " + str(cap[p])}, main game '
              f'{"clear" if main[p]["ok"] else "FAIL " + str(main[p])}')
    missing = [b for b in BOARDS if b not in book]
    print(f'\n{len(book) - bad}/{len(BOARDS)} boards have a route that clears from the main game'
          + (f'; no route for {missing}' if missing else ''))
    sys.exit(1 if bad or missing else 0)


def cmd_status(a):
    import demos
    book = load()
    his = {b: len(demos.load_demo(p)[0]) for b, p in his_recordings().items()}
    n_c = {}
    for p in glob.glob(os.path.join(CAND_DIR, 'b*.npz')):
        b = int(os.path.basename(p)[1:3])
        n_c[b] = n_c.get(b, 0) + 1
    print(f'{"board":>5s} {"route":>6s} {"his":>6s} {"gain":>6s}  {"candidates":>10s}  source')
    for b in BOARDS:
        r = book.get(b)
        h = his.get(b)
        gain = f'{h - r["frames"]:+d}' if r and h else '-'
        print(f'{b:5d} {r["frames"] if r else "-":>6} {h or "-":>6} {gain:>6s}  {n_c.get(b, 0):>10d}  '
              f'{r["source"] if r else "NO ROUTE"}')
    beat = [b for b in BOARDS if b in book and b in his and book[b]['frames'] < his[b]]
    print(f'\nroutes: {len(book)}/14 (guaranteed emeralds); faster than Tyler: {len(beat)}/14 {beat}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', nargs='?', default='status', choices=['status', 'seed', 'harvest', 'verify'])
    ap.add_argument('--run', default='d9', help='the train_ss_multistage run whose env settings replay')
    ap.add_argument('--force', action='store_true', help='seed: replace routes already in the book')
    ap.add_argument('--every', type=float, help='harvest: repeat every this many minutes')
    a = ap.parse_args()
    dict(status=cmd_status, seed=cmd_seed, harvest=cmd_harvest, verify=cmd_verify)[a.cmd](a)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.path.insert(0, HOME)
        args = json.loads(sys.argv[3])
        dict(captured=_run_captured, main=_run_main)[sys.argv[2]](args['paths'], args['run'])
    else:
        main()
