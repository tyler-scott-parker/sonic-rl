import json, sys, collections
log, cur = sys.argv[1], json.load(open(sys.argv[2]))
ptr = {k.split(':')[-1]: v['ptr'] for k, v in cur.items()}
eps = collections.defaultdict(list)
for line in open(log):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    eps[r['act']].append(r)
share = lambda r: r.get('boss_share') or r.get('ring_share') or r.get('front_share')
cell = lambda r: (r['x'] // 256 * 256, r['y'] // 256 * 256)
for act, rs in eps.items():
    p = ptr.get(act)
    fresh = [r for r in rs if r['start'] == 0 and not share(r)][-300:]
    at = [r for r in rs if p is not None and p <= r['start'] <= p + 4 and not share(r)][-500:]
    print(f"\n{act} (pointer {p})")
    for name, L in (('fresh starts', fresh), ('at the pointer', at)):
        if not L:
            continue
        out = collections.Counter(r['outcome'] for r in L)
        ends = [r for r in L if r['outcome'] in ('stuck', 'death', 'timeout')]
        cl = collections.Counter(cell(r) for r in ends).most_common(3)
        causes = collections.Counter(r.get('cause', '-') for r in L if r['outcome'] == 'death')
        rings0 = sum(1 for r in L if r['outcome'] == 'death' and r['rings'] == 0 and r.get('hurt'))
        prog = sorted(r['progress'] for r in L)
        print(f"  {name} ({len(L)}): {dict(out.most_common())}; progress median {prog[len(prog)//2]:.1%}; "
              f"deaths by cause {dict(causes)}, after a hit with 0 rings {rings0}")
        print(f"    where non-clears end (256-px cells): " + ', '.join(f"{c} {n} ({n / max(1, len(ends)):.0%})" for c, n in cl))
