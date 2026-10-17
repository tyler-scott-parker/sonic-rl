#!/usr/bin/env python3
"""check_fight_lead.py — does MG_FIGHT_LEAD (RUNS 'fight_lead') extend the three fight rules (boss share, fight
bar, imitation 'fights') to pointers / start points just before one of his fights, and change nothing at 0?

    python3 check_fight_lead.py [RUN] [LEAD]          # default mg2 600

1. lead 0 = the old rules exactly, for every start point of every act: fight_at = 'inside a fight';
   boss_ranges (pointer set to each start point in turn) = the old code; the bar's slack = the old rule
2. lead LEAD: a pointer inside a fight gets the same range as before; one up to LEAD frames before a fight
   gets that fight, lo = its first start point, hi = lo + DEMO_BAND; one further before gets none
3. lead LEAD: the bar's slack is MG_FIGHT_SLACK for start points up to LEAD frames before a fight (and inside
   one as before), MATCH_SLACK elsewhere
4. the imitation pull (front off, to compare like with like): lead 0 = before; lead LEAD adds his frames of
   the fight ahead for acts whose pointer is in the lead-in, nothing else
5. now, with the current pointers: which acts the lead brings in, and what their at-pointer tries would
   have matched under the fight bar (the run's log)
CPU only, no model; the env's episode log goes to a temp file. Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    lead = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    sys.path.insert(0, HOME)
    import numpy as np
    import train_main as T
    T.configure_env(run)
    real_log = os.environ['MG_LOG']
    os.environ.update(MG_LOG=tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name,
                      CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    print(f"RUNS {run} fight_lead={T.RUNS[run].get('fight_lead', 0)} (checking lead {lead}); "
          f"MG_FIGHT_SLACK {C.MG_FIGHT_SLACK}, DEMO_BAND {C.DEMO_BAND}")
    env = C.SonicCoordinator(acts=T.RUNS[run]['acts'], demo_prob=0.0)
    infos = env.demo_info()
    cur = json.load(open(T.paths(run)['curriculum']))
    segs = env._segs

    def old_ranges():                                   # boss_ranges as it was before the lead (git 0c28ac8)
        out = []
        for s in segs:
            p = env._ptr.get(s['key'])
            if p is None:
                continue
            last = len(s['ckpts']) - 1
            p = min(p, last)
            t = s['ckpts'][p]['t']
            for a, b in s['route']['fights']:
                if a <= t < b:
                    lo = next(k for k, c in enumerate(s['ckpts']) if c['t'] >= a)
                    out.append((s['key'], lo, min(last, p + C.DEMO_BAND)))
                    break
        return out

    def new_ranges():
        return [(s['key'], lo, hi) for s, lo, hi in env.boss_ranges()]

    def slack(s, ck):
        env._seg, env._ck = s, ck
        return env._slack()

    old_slack = lambda ck: C.MG_FIGHT_SLACK if C.MG_FIGHT_SLACK > 0 and ck.get('boss') else C.MATCH_SLACK

    print('\n1. lead 0 = the old rules')
    C.MG_FIGHT_LEAD = 0
    n = same_f = same_r = same_s = 0
    for s in segs:
        for p, ck in enumerate(s['ckpts']):
            n += 1
            inside = next(((a, b) for a, b in s['route']['fights'] if a <= ck['t'] < b), None)
            same_f += C.fight_at(s['route']['fights'], ck['t']) == inside
            env._ptr = {s['key']: p}
            same_r += new_ranges() == old_ranges()
            same_s += slack(s, ck) == old_slack(ck)
    check(same_f == n, f'fight_at = inside a fight at all {n:,} start points')
    check(same_r == n, f'boss_ranges = the old code with the pointer at each of the {n:,} start points')
    check(same_s == n, f"the bar's slack = the old rule at all {n:,} start points")

    print(f'\n2. lead {lead}: boss_ranges, pointer at every start point')
    C.MG_FIGHT_LEAD = lead
    ok_in = ok_lead = ok_out = n_in = n_lead = n_out = 0
    for s in segs:
        last = len(s['ckpts']) - 1
        for p, ck in enumerate(s['ckpts']):
            t = ck['t']
            env._ptr = {s['key']: p}
            got = new_ranges()
            inside = next(((a, b) for a, b in s['route']['fights'] if a <= t < b), None)
            ahead = next(((a, b) for a, b in s['route']['fights'] if a - lead <= t < a), None)
            if inside:
                n_in += 1
                C.MG_FIGHT_LEAD = 0
                ok_in += got == new_ranges()
                C.MG_FIGHT_LEAD = lead
            elif ahead:
                n_lead += 1
                lo = next(k for k, c in enumerate(s['ckpts']) if c['t'] >= ahead[0])
                ok_lead += got == [(s['key'], lo, min(last, lo + C.DEMO_BAND))]
            else:
                n_out += 1
                ok_out += got == []
    check(ok_in == n_in, f'{n_in:,} pointers inside a fight: the same range as at lead 0')
    check(ok_lead == n_lead and n_lead > 0, f'{n_lead:,} pointers in a lead-in: that fight, its first start point + band')
    check(ok_out == n_out, f'{n_out:,} pointers elsewhere: none')

    print(f'\n3. lead {lead}: the bar')
    ok = tot = 0
    for s in segs:
        for ck in s['ckpts']:
            t = ck['t']
            near = any(a - lead <= t < b for a, b in s['route']['fights'])
            want = C.MG_FIGHT_SLACK if C.MG_FIGHT_SLACK > 0 and (ck.get('boss') or near) else C.MATCH_SLACK
            tot += 1
            ok += slack(s, ck) == want
    check(ok == tot, f"slack {C.MG_FIGHT_SLACK} in and up to {lead} frames before a fight, {C.MATCH_SLACK} elsewhere ({tot:,} start points)")
    for s in segs:                                      # the boundary, per act's first fight
        for a, b in s['route']['fights']:
            ks = [k for k, c in enumerate(s['ckpts']) if a - lead <= c['t'] < a]
            if ks:
                before = ks[0] - 1
                print(f"    {s['name']}: fight t {a}..{b}; lead-in start points {ks[0]}..{ks[-1]}"
                      + (f"; {before} (t {s['ckpts'][before]['t']}) keeps {slack(s, s['ckpts'][before])}" if before >= 0 else ''))

    print(f'\n4. lead {lead}: the imitation pull (front off)')
    fights = {d['key']: d['fights'] for d in infos}
    cfg = {**T.IMITATION_DEFAULTS, **T.RUNS[run]['imitation'], 'front': False, 'fights': True}
    cb = T.ImitationCallback(cfg, run, T.paths(run)['curriculum'])
    cb.load_data()
    cb._fights = fights
    C.MG_FIGHT_LEAD = 0
    at0 = set(cb.frontier().tolist())
    C.MG_FIGHT_LEAD = lead
    atL = set(cb.frontier().tolist())
    S = cb.S
    extra, acts = set(), []
    for key, idx in cb.keys.items():
        t0 = cur[key]['frac'] * int(S['T'][idx[0]])
        ahead = next(((a, b) for a, b in fights.get(key, ()) if a - lead <= t0 < a), None)
        if ahead:
            t = S['t'][idx]
            w = idx[(t >= t0 - cfg['behind']) & (t <= ahead[1])]
            extra |= set(w.tolist())
            acts.append((key.split(':')[-1], ahead, len(w)))
    check(atL == at0 | extra, f'{len(at0):,} -> {len(atL):,} focused frames: + the fights ahead of '
          + (', '.join(f'{a} (t {f[0]}..{f[1]}, {n:,} frames)' for a, f, n in acts) or 'no act'))

    print('\n5. now: acts the lead brings in, and their at-pointer tries')
    env._ptr = {k: v['ptr'] for k, v in cur.items()}
    C.MG_FIGHT_LEAD = 0
    r0 = {k for k, _, _ in new_ranges()}
    C.MG_FIGHT_LEAD = lead
    rL = new_ranges()
    print('    boss share at lead 0: ' + (', '.join(sorted(k.split(':')[-1] for k in r0)) or 'none'))
    print(f'    boss share at lead {lead}: ' + ', '.join(f"{k.split(':')[-1]} {lo}..{hi}" for k, lo, hi in rL))
    rows = [json.loads(l) for l in open(real_log) if l.startswith('{')]
    for k, lo, hi in rL:
        if k in r0:
            continue
        act, p = k.split(':')[-1], cur[k]['ptr']
        at = [r for r in rows if r['act'] == act and r['start'] == p and not r.get('boss_share')
              and not r.get('ring_share') and not r.get('front_share')]
        cl = [r for r in at if r['outcome'] == 'clear']
        tg = at[0]['target'] if at else 0
        print(f"    {act} pointer {p}: {len(at)} tries there, {len(cl)} clears; matched at 1.25x "
              f"{sum(r['frames'] <= (1 + C.MATCH_SLACK) * r['target'] + C.MATCH_ADD for r in cl)}, at the fight bar "
              f"{sum(r['frames'] <= (1 + C.MG_FIGHT_SLACK) * r['target'] + C.MATCH_ADD for r in cl)} (his {tg} frames)")
    print('\n' + ('ALL CHECKS PASS' if not bad else f'{len(bad)} FAILED: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
