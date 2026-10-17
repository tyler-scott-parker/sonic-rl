#!/usr/bin/env python3
"""forget_check.py — per-act / per-board forgetting alarm for a running run (Tyler 2026-10-16 ~21:50: "Do we
have any protections against catastrophic forgetting?" ... "yes" to adding a check to the overnight watch).

    forget_check.py RUN [--state FILE]      # mg2 (mg_log_<run>.jsonl) or d10 (ss_log_<run>.jsonl)
    forget_check.py RUN --backtest [HOURS]  # replay the whole log, checking every HOURS (default 0.5)

The collapse checks (d9_status: all demo starts together; mg_status: entropy) miss one act sliding while the
rest hold: AIZ1's clears fell 60-74% -> 14-20% on 2026-10-15 and only a read of the logs found it. This
looks at fresh starts only (from the act's / board's own start: the target never moves, unlike the
curriculum's start points, so a drop is the policy, not the bar), per act or board, the last N_NOW tries vs
the N_BEFORE before them, on two measures:
  clears    the clear rate, where the earlier rate was >= MIN_RATE: ALERT when it falls to <= RATIO of it
            and a two-proportion z >= Z (normal approximation);
  reach     how far into the route a try gets (mg: progress along his route; ss: spheres / board total):
            ALERT when the median falls to <= RATIO of the earlier median and a rank-sum z >= Z.
Windows count tries, not time, so a rarely drawn act is judged on as many tries as a common one (it just
takes longer to fill). With --state an alert is printed once; it re-arms when that measure recovers to
>= RECOVER of the earlier level, or fires again if it halves once more."""
import collections
import json
import math
import os
import statistics
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
N_NOW, N_BEFORE = 50, 150
MIN_RATE = 0.15
RATIO = 0.5
Z = 3.0
RECOVER = 0.75


def load(run):
    """(time, key, cleared, reach) per fresh try, in log order."""
    mg = run.startswith('mg')
    path = os.path.join(REPO, f'mg_log_{run}.jsonl' if mg else f'ss_log_{run}.jsonl')
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if mg:
                if r.get('start') != 0 or r.get('ring_share') or r.get('front_share') or r.get('boss_share'):
                    continue
                out.append((r.get('t', 0), r['act'], r.get('outcome') == 'clear', float(r.get('progress') or 0)))
            else:
                if r.get('demo') or 'stage' not in r:
                    continue
                out.append((r.get('t', 0), int(r['stage']), bool(r.get('success')),
                            r.get('spheres', 0) / max(1, r.get('stage_total') or 1)))
    return out


def rank_z(before, now):
    """Mann-Whitney rank-sum z (positive = `now` ranks lower), normal approximation with tie correction."""
    both = sorted((v, i) for i, v in enumerate(before + now))
    ranks, i = [0.0] * len(both), 0
    ties = 0.0
    while i < len(both):
        j = i
        while j + 1 < len(both) and both[j + 1][0] == both[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[both[k][1]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    n1, n2 = len(before), len(now)
    n = n1 + n2
    u = sum(ranks[n1:]) - n2 * (n2 + 1) / 2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    return (n1 * n2 / 2 - u) / math.sqrt(var) if var > 0 else 0.0


def judge(tries):
    """The two measures for one act's fresh tries: {measure: (before, now, z, falling)}, {} if too few."""
    if len(tries) < N_NOW + N_BEFORE:
        return {}
    before, now = tries[-(N_NOW + N_BEFORE):-N_NOW], tries[-N_NOW:]
    res = {}
    pb, pn = sum(c for c, _ in before) / N_BEFORE, sum(c for c, _ in now) / N_NOW
    if pb >= MIN_RATE:
        p = (pb * N_BEFORE + pn * N_NOW) / (N_BEFORE + N_NOW)
        se = math.sqrt(p * (1 - p) * (1 / N_BEFORE + 1 / N_NOW))
        z = (pb - pn) / se if se > 0 else 0.0
        res['clears'] = (pb, pn, z, pn <= RATIO * pb and z >= Z)
    mb, mn = statistics.median(r for _, r in before), statistics.median(r for _, r in now)
    if mb > 0:
        z = rank_z([r for _, r in before], [r for _, r in now])
        res['reach'] = (mb, mn, z, mn <= RATIO * mb and z >= Z)
    return res


def check(per, st, quiet=False):
    """ALERT lines (and, unless quiet, a status line per act); `st` = {f'{key}/{measure}': level at the alert}."""
    alerts = []
    for key in sorted(per):
        res = judge(per[key])
        bits = []
        for m, (b, n, z, falling) in res.items():
            bits.append(f'{m} {b:.0%} -> {n:.0%} (z {z:+.1f})')
            sk = f'{key}/{m}'
            if sk in st and n >= RECOVER * st[sk]['before']:
                del st[sk]                                   # recovered: re-arm
            if falling and (sk not in st or n <= RATIO * st[sk]['now']):
                alerts.append(f'ALERT forgetting? {key} fresh-start {m}: {b:.0%} over the {N_BEFORE} tries before '
                              f'-> {n:.0%} over the last {N_NOW} (z {z:.1f})')
                st[sk] = dict(before=st.get(sk, {}).get('before', b), now=n)
        if not quiet:
            print(f'  {key}: {len(per[key])} fresh tries; ' + ('; '.join(bits) if bits else
                  f'(needs {N_NOW + N_BEFORE} to judge)'))
    return alerts


def main():
    args = sys.argv[1:]
    run = args[0] if args and not args[0].startswith('-') else 'mg2'
    tries = load(run)
    if '--backtest' in args:
        i = args.index('--backtest')
        every = float(args[i + 1]) * 3600 if len(args) > i + 1 else 1800
        per, st, nxt, fired = collections.defaultdict(list), {}, None, 0
        for t, key, c, r in tries:
            if nxt is None:
                nxt = t + every
            while t >= nxt:
                for a in check(per, st, quiet=True):
                    fired += 1
                    print(f'{__import__("datetime").datetime.fromtimestamp(nxt):%m-%d %H:%M} {a}')
                nxt += every
            per[key].append((c, r))
        print(f'{run} backtest: {len(tries):,} fresh tries, {fired} alerts')
        return
    per = collections.defaultdict(list)
    for _, key, c, r in tries:
        per[key].append((c, r))
    sp = args[args.index('--state') + 1] if '--state' in args else None
    st = json.load(open(sp)) if sp and os.path.exists(sp) else {}
    print(f'{run}: forgetting check (fresh starts, last {N_NOW} vs the {N_BEFORE} before)')
    alerts = check(per, st)
    for a in alerts:
        print(a)
    if sp:
        with open(sp, 'w') as f:
            json.dump(st, f)


if __name__ == '__main__':
    main()
