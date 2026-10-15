#!/usr/bin/env python3
"""check_boss_hits.py — does a slot-free rule find every boss hit in Tyler's
full-game recording, and nothing else?

find_boss_hits.py found each boss's hits-left count at +$29 of its object
(collision_property: S3K's touch code decrements it instead of destroying an
enemy that has one). The env can't know which slot a boss is in, so it
needs a rule: every agent step (4 frames), a hit = an object slot (2..109)
occupied at both steps whose +$29 is exactly 1 lower than at the last step,
and whose +$28 (collision_flags)
  strict: was an enemy's at the last step (nonzero, type bits $C0 clear);
  middle: was nonzero at either step (any touchable object);
  loose:  anything;
  boss:   per slot, over time (BossHits): the count it drops from has been
          BOSS_MIN..BOSS_MAX since the object appeared (bosses: 6, 8, 9, 14;
          3-hit badniks: 3; Robotnik's unkillable drill car in MGZ2 and the
          AIZ1 fire scene's object: 255 and 96), +$28 was nonzero within TOUCH_RECENT frames (a hit
          boss flashes untouchable, so "at the last step" misses the next
          hit), and >= MIN_GAP frames since that slot's last hit.
(The scan showed why strict may not do: LRZ3's boss is "hurt" type, $B8,
and FBZ1/HPZ1/DEZ2's read 0 on the frame before the count dropped. The first
run of this check: strict/middle found 123/137 of 183 hits; loose all 183 but
183 more outside fights, 136 of them an AIZ2 timer object at +$29.)

Replays the recording, applies the rules every FRAME_SKIP frames, and compares
with the scan's +$29 counters (logs/boss_hits_scan.json): per fight, the
counter's steps found within a step; and every hit outside a fight window
(false alarms, unless a boss the windows missed).

    python3 attic/one_off_tools/check_boss_hits.py
"""
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
FIRST = 2                                                     # 0 and 1: Sonic and Tails


def objects(ram):
    return ram[OBJ0:OBJ0 + SLOTS * SLOT_LEN].reshape(SLOTS, SLOT_LEN)[FIRST:]


RULES = ('strict', 'middle', 'loose', 'boss')
BOSS_MIN, BOSS_MAX, TOUCH_RECENT, MIN_GAP = 5, 16, 64, 16


class BossHits:
    """The 'boss' rule, fed one object table per agent step."""

    def __init__(self):
        n = SLOTS - FIRST
        self.maxseen = np.zeros(n, int)
        self.touch = np.full(n, -10 ** 9)
        self.last = np.full(n, -10 ** 9)

    def step(self, frame, prev, cur):
        occ = cur[:, :4].any(1)
        gone = ~occ | ~prev[:, :4].any(1)
        self.maxseen[gone] = 0
        self.touch[gone] = -10 ** 9
        self.last[gone] = -10 ** 9
        v = cur[:, 0x29].astype(int)
        self.maxseen = np.maximum(self.maxseen, np.where(occ, v, 0))
        self.touch[(prev[:, 0x28] != 0) | (cur[:, 0x28] != 0)] = frame
        hit = (occ & ~gone & (v == prev[:, 0x29].astype(int) - 1)
               & (self.maxseen >= BOSS_MIN) & (self.maxseen <= BOSS_MAX)
               & (frame - self.touch <= TOUCH_RECENT) & (frame - self.last >= MIN_GAP))
        self.last[hit] = frame
        return np.flatnonzero(hit)


def hits(prev, cur, frame, boss):
    """{rule: slots (index from FIRST) hit between two object tables}."""
    step = prev[:, :4].any(1) & cur[:, :4].any(1) & (cur[:, 0x29].astype(int) == prev[:, 0x29].astype(int) - 1)
    enemy = (prev[:, 0x28] != 0) & ((prev[:, 0x28] & 0xC0) == 0)
    touchable = (prev[:, 0x28] != 0) | (cur[:, 0x28] != 0)
    return {'strict': np.flatnonzero(step & enemy), 'middle': np.flatnonzero(step & touchable),
            'loose': np.flatnonzero(step), 'boss': boss.step(frame, prev, cur)}


def main():
    scan = json.load(open('logs/boss_hits_scan.json'))
    actions, meta = demos.load_play(DEMO)
    table = meta['action_table']
    acts = [(a['name'], a['start'], a['clear']) for a in meta['acts'] if a['clear'] is not None]
    clear_of = {n: c for n, s, c in acts}
    windows = []                                              # (act, first, last frame, counter steps)
    for w in scan:
        c = clear_of[w['act']]
        main = [cand for cand in w['candidates'] if cand['off'] == 0x29][:1]   # the boss's own counter
        steps = sorted(f + c for cand in main for f in cand['at'])
        windows.append((w['act'], w['first'] + c, w['last'] + c, steps))
    env = C.make_retro()
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    swap, prev, boss = None, None, BossHits()
    found = {r: collections.defaultdict(list) for r in RULES}     # window index -> hit frames
    stray = {r: [] for r in RULES}                                # (frame, act, slot, +$29 after, +$28, code)
    t0 = time.time()
    for i, a in enumerate(actions):
        _, info = C.emu_frame(env, table[a])
        if i % C.FRAME_SKIP:
            continue
        raw = np.frombuffer(bytes(env.get_ram()), dtype=np.uint8)
        if swap is None:
            try:
                swap = not np.array_equal(C._ram_words(env, info), raw)
            except RuntimeError:
                continue
        ram = raw.reshape(-1, 2)[:, ::-1].reshape(-1) if swap else raw
        cur = objects(ram).copy()
        if prev is not None:
            w = next((k for k, (_, f0, f1, _) in enumerate(windows) if f0 <= i <= f1), None)
            for rule, slots in hits(prev, cur, i, boss).items():
                for s in slots:
                    if w is not None:
                        found[rule][w].append(i)
                    else:
                        act = next((n for n, s0, c in acts if s0 <= i <= c), '-')
                        stray[rule].append((i, act, int(s) + FIRST, int(cur[s, 0x29]), f'{int(prev[s, 0x28]):02X}',
                                            f'{int.from_bytes(bytes(cur[s, 0:4]), "big"):06X}'))
        prev = cur
        if i % 100000 == 0:
            print(f'[frame {i:,}, {time.time() - t0:.0f} s]', flush=True)
    print(f"\nper fight window: the counter's hits; per rule, found of them / all the rule fired")
    print(f"{'act':6s} {'hits':>5s}  " + '  '.join(f'{r:>13s}' for r in RULES))
    tot = collections.Counter()
    for k, (act, f0, f1, steps) in enumerate(windows):
        cells = []
        for r in RULES:
            got = found[r].get(k, [])
            m = sum(any(0 <= g - s < C.FRAME_SKIP for g in got) for s in steps)
            tot[r, 'm'] += m
            tot[r, 'n'] += len(got)
            cells.append(f'{m:>4d} / {len(got):<4d}')
        tot['c'] += len(steps)
        print(f'{act:6s} {len(steps):5d}  ' + '  '.join(f'{c:>13s}' for c in cells))
    print(f"{'all':6s} {tot['c']:5d}  " + '  '.join(f"{tot[r, 'm']:>4d} / {tot[r, 'n']:<4d}" for r in RULES))
    for r in RULES:
        print(f'\n{r}: hits outside every fight window: {len(stray[r])}')
        by_act = collections.Counter(row[1] for row in stray[r])
        print('   by act: ' + ', '.join(f'{a} {n}' for a, n in by_act.most_common()))
        for row in stray[r][:12]:
            print('   frame {:,} {}: slot {} -> {} left, +$28 {}, code {}'.format(*row))


if __name__ == '__main__':
    main()
