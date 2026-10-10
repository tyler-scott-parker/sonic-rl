"""First decision at (26,21) per episode: what does the policy choose?
Yardstick, d3's 537K save from 56% (steer_episodes_v4_ptr.json): 87 eps reach it,
first choice jump 74 / left 11 / straight 2, median P(jump) 0.998; with (25,21)
still blue: 65 of 65 jump. Usage: pocket_choice.py steer_episodes_<tag>.json"""
import json, sys
from collections import Counter
from statistics import median
NAMES = ('straight', 'left', 'right', 'jump')
eps = json.load(open(sys.argv[1]))
rows = []                                    # (code of (25,21), decision)
for e in eps:
    ds = [d for d in e['decisions'] if d['cell'] == [26, 21]]
    if ds:
        boards = sorted(e['boards'], key=lambda b: b['i'])
        g = [b for b in boards if b['i'] <= ds[0]['i']][-1]['grid']
        rows.append((g[25][21], ds[0]))
def show(label, sel):
    if not sel:
        print(f'  {label}: none'); return
    c = Counter(NAMES[d['a']] for d in sel)
    print(f'  {label}: {len(sel)} eps | chose ' + ', '.join(f'{k} {v}' for k, v in c.most_common())
          + ' | median P ' + ', '.join(f'{k} {median(d["p"][i] for d in sel):.3f}' for i, k in enumerate(NAMES)))
print(f'{len(eps)} episodes, {len(rows)} reach (26,21)')
show('all first visits', [d for _, d in rows])
show('(25,21) still BLUE (pass = left is the majority)', [d for code, d in rows if code == 2])
show('(25,21) already red', [d for code, d in rows if code == 1])
