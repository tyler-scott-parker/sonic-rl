"""First decision at (26,21) with (25,21) blue: heading on arrival, the choice, and the next cell Sonic
actually enters. Cell = ((y+0x80)>>8)%32, ((x+0x80)>>8)%32 (context doc). Usage: pocket_next.py <steer_episodes json>"""
import json, sys
from collections import Counter
NAMES = ('straight', 'left', 'right', 'jump')
cell = lambda fr: (((fr['y'] + 0x80) >> 8) % 32, ((fr['x'] + 0x80) >> 8) % 32)
eps = json.load(open(sys.argv[1]))
rows = Counter(); onto = Counter(); arrive = Counter()
for e in eps:
    ds = [d for d in e['decisions'] if d['cell'] == [26, 21]]
    if not ds:
        continue
    d = ds[0]
    boards = sorted(e['boards'], key=lambda b: b['i'])
    if [b for b in boards if b['i'] <= d['i']][-1]['grid'][25][21] != 2:
        continue
    fr = e['frames']
    # frames carry absolute 'f'; decisions carry an index into frames
    prev = [cell(f) for f in fr[:d['i']] if cell(f) != (26, 21)]
    came = prev[-1] if prev else None
    nxt = next((cell(f) for f in fr[d['i']:] if cell(f) != (26, 21)), None)
    arrive[came] += 1
    rows[(NAMES[d['a']], nxt)] += 1
    onto[nxt == (25, 21)] += 1
print(f'{sum(onto.values())} first visits with (25,21) blue')
print('  arrived from:', dict(arrive))
print('  (choice, next cell):', dict(rows.most_common()))
print(f'  stepped onto (25,21) next: {onto[True]} of {sum(onto.values())}')
