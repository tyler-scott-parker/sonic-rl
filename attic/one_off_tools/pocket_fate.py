"""What happens to blue (25,21): collected (from which cell), and what's left blue when episodes stall.
Usage: pocket_fate.py <steer_episodes json>"""
import json, sys
from collections import Counter
cell = lambda fr: (((fr['y'] + 0x80) >> 8) % 32, ((fr['x'] + 0x80) >> 8) % 32)
eps = json.load(open(sys.argv[1]))
out = Counter(); entry = Counter(); last_blue = Counter(); by_out = Counter()
for e in eps:
    o = 'cleared' if e['cleared'] else 'stall' if e['stalled'] else 'red/other'
    out[o] += 1
    fin = sorted(e['boards'], key=lambda b: b['i'])[-1]['grid']
    got = fin[25][21] != 2
    by_out[(o, 'got (25,21)' if got else '(25,21) still blue')] += 1
    if got:
        cs = [cell(f) for f in e['frames']]
        j = next((k for k, c in enumerate(cs) if c == (25, 21)), None)
        if j:
            entry[next((c for c in reversed(cs[:j]) if c != (25, 21)), None)] += 1
    if o == 'stall':
        blues = [(r, c) for r in range(32) for c in range(32) if fin[r][c] == 2]
        if len(blues) <= 3:
            last_blue[tuple(blues)] += 1
print('outcomes:', dict(out))
print('by outcome:', dict(by_out))
print('entered (25,21) from:', dict(entry))
print('stalls with <= 3 blues left, which:', dict(last_blue.most_common(6)))
