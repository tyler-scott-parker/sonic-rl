#!/usr/bin/env python3
"""audit_play.py — check a base-game recording act by act after each session.

  python3 audit_play.py                         # the newest demos/play_*.npz
  python3 audit_play.py demos/play_<stamp>.npz

Replays the playthrough (it must reproduce every recorded act exactly), then
lists each act — start and clear frame, its time by the recording's own count
and by the in-game timer (its reading at the clear, and the time it ran since
the act began — they differ by a few lag frames, more after a death, which
puts the clock back to the starpost), deaths, special/bonus stages, boss fights — and
flags what needs a human look. The act-clear rule is "the in-game timer stops
with Sonic alive in the level", and its known ways to go wrong are:
  - a cutscene that stops the timer mid-act reads as a clear (the act then
    "starts" again under the same name);
  - an act 2 ends at the capsule, which no recording has shown yet: an act 2
    that never clears, or clears with no boss fight before it, is suspect;
  - acts out of the game's order, or implausibly short.
Nothing here changes the file. Read the flags against what you saw on screen.
"""
import glob
import os
import sys

import numpy as np

import coordinator as C
import demos

# Sonic & Tails' order through S3&K (internal zone ids differ: FBZ is 4, ICZ 5).
ORDER = ['AIZ', 'HCZ', 'MGZ', 'CNZ', 'ICZ', 'LBZ', 'MHZ', 'FBZ', 'SOZ', 'LRZ',
         'HPZ', 'SSZ', 'DEZ', 'DDZ']
SHORT = 15 * 60          # an act under 15 s gets a look


def fmt(frames):
    return f'{frames // 3600}:{frames // 60 % 60:02d}.{round(frames % 60 * 100 / 60):02d}'


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(os.path.join(demos.DEMO_DIR, 'play_*.npz')), key=os.path.getmtime)
    actions, meta = demos.load_play(path)
    print(f'{os.path.basename(path)}: {len(actions):,} frames ({fmt(len(actions))}), '
          f'{len(meta.get("sessions", []))} session(s), {meta.get("rewinds", 0)} rewinds, '
          f'from {meta["start_state"]}')

    boss = []                # (start, end) frames with Boss_flag up
    cur = {'b': None}
    clock = np.zeros(len(actions), dtype=np.int64)   # in-game timer, in frames

    def on_frame(i, info):
        m, sec, fr = C.game_time(info)
        clock[i] = m * 3600 + sec * 60 + fr
        up = int(info['mg_boss']) != 0
        if up and cur['b'] is None:
            cur['b'] = i
        elif not up and cur['b'] is not None:
            boss.append((cur['b'], i))
            cur['b'] = None

    split = demos.replay_play(actions, meta, on_frame=on_frame)
    if cur['b'] is not None:
        boss.append((cur['b'], len(actions)))
    got = [(a['name'], a['start'], a['clear']) for a in split.summary()]
    want = [(a['name'], a['start'], a['clear']) for a in meta['acts']]
    if got != want:
        print(f'!! REPLAY MISMATCH: recorded {want}\n   replay   {got}')
        sys.exit(1)
    print('replay reproduces every recorded act\n')

    flags = []
    acts = split.summary()
    print(f'{"act":5s} {"start":>8s} {"clear":>8s} {"own time":>9s} {"in-game":>8s} {"elapsed":>8s} '
          f'{"deaths":>6s} {"stages":>6s}  boss fights (frames into the act)')
    for k, a in enumerate(acts):
        fights = [(s - a['start'], e - a['start']) for s, e in boss
                  if a['start'] <= s and (a['clear'] is None or s <= a['clear'])]
        own = fmt(a['t']) if a['clear'] is not None else f'({fmt(a["t"])})'
        # the in-game clock from the act's first frame to its clear (a start
        # state can begin with time on it: AIZ1's reads 0:01.15)
        el = int(clock[a['clear']] - clock[a['start']]) if a['clear'] is not None else None
        print(f'{a["name"]:5s} {a["start"]:8d} {str(a["clear"]):>8s} {own:>9s} '
              f'{a["time"] or "-":>8s} {fmt(el) if el is not None else "-":>8s} '
              f'{len(a["deaths"]):6d} {len(a["stages"]):6d}  '
              f'{fights if fights else "-"}')
        last = k == len(acts) - 1
        if a['clear'] is None and not last:
            flags.append(f'{a["name"]}: never cleared, yet another act follows')
        if a['clear'] is not None:
            if abs(a['t'] - el) > 120 and not a['deaths'] and not a['stages']:
                flags.append(f'{a["name"]}: own count {fmt(a["t"])} vs in-game {fmt(el)} elapsed, with no '
                             f'death or stage to explain it — did the timer stop and restart mid-act?')
            if a['t'] < SHORT:
                flags.append(f'{a["name"]}: cleared after only {fmt(a["t"])} — a cutscene timer stop?')
            if a['name'].endswith('2') and not fights:
                flags.append(f'{a["name"]}: an act 2 cleared with no boss fight before it — '
                             f'capsule? check the clear frame {a["clear"]}')
            if fights and fights[-1][1] < a['t'] - 30 * 60:
                flags.append(f'{a["name"]}: cleared {fmt(a["t"] - fights[-1][1])} after its last '
                             f'boss fight ended — check what stopped the timer')
        if k and a['name'] == acts[k - 1]['name']:
            flags.append(f'{a["name"]} follows itself: the first one\'s "clear" at frame '
                         f'{acts[k - 1]["clear"]} was probably a cutscene stopping the timer')
        if k:
            z0, z1 = acts[k - 1]['name'][:-1], a['name'][:-1]
            if z0 in ORDER and z1 in ORDER and ORDER.index(z1) < ORDER.index(z0):
                flags.append(f'{a["name"]} comes after {acts[k - 1]["name"]}: out of the game\'s order')
            if a['name'][:-1] not in ORDER:
                flags.append(f'{a["name"]}: zone not in the known list')
    print()
    if flags:
        print(f'{len(flags)} thing(s) to look at:')
        for f in flags:
            print(f'  - {f}')
    else:
        print('nothing flagged')
    done = [a for a in acts if a['clear'] is not None]
    print(f'\n{len(done)} act(s) cleared; total act time {fmt(sum(a["t"] for a in done))} by the '
          f'recording\'s own count (deaths count, special/bonus stages don\'t)')


if __name__ == '__main__':
    main()
