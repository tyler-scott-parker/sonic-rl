"""Watch numbers for a main-game run (train_main.py), e.g. mg1.

    python3 attic/one_off_tools/mg_status.py mg1 [--state logs/mg1_watch_state.json]

Per act: start point (curriculum file), outcomes of the last 500 episodes,
fresh starts (start point 0) and their clears vs Tyler's frames, giant-ring
exits (`stage`, the watch item: an exit the value net likes could become a way
to end bad episodes). Steps = the trainer's last total_timesteps.
ALERT lines (mg1's proposed criteria, doc "Run mg1"):
  - stall: a start point unmoved for 3M steps (needs --state to remember moves)
  - wall gate: 5M steps and AIZ1 still at or behind start point 69 (82%)
"""
import argparse
import collections
import json
import os
import re
import sys
import time

REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO)
from mg_stones import since_arrival, next_point_works        # noqa: E402  (one rule for both tools)
STALL_STEPS = 3_000_000
STONES_AT = 200          # episodes started AT one start point without moving it: stepping stones (mg2)
ENTROPY_MAX = 1.2        # policy entropy (max ln 9 = 2.2): mg2 collapsed as it rose 0.6 -> 1.8
GATE_STEPS, GATE_PTR = 5_000_000, 69


def steps_of(run):
    last = ent = None
    try:
        with open(os.path.join(REPO, 'logs', f'main_{run}.log'), errors='replace') as f:
            for line in f:
                m = re.search(r'total_timesteps\s*\|\s*(\d+)', line)
                if m:
                    last = int(m.group(1))
                m = re.search(r'entropy_loss\s*\|\s*(-?[\d.]+)', line)
                if m:
                    ent = -float(m.group(1))
    except FileNotFoundError:
        pass
    return last, ent


def s_get(st, key, name):
    return st.get(key, {}).get(name)


def fresh_frontier(fresh, n=32, min_n=8):
    """coordinator.fresh_frontier's rule (FRONT_N, FRONT_MIN_N; importing it would load the emulator):
    how far into the act (his frames) the median of the last n fresh starts got; None under min_n."""
    reach = sorted(r['progress'] * r['target'] for r in fresh[-n:])
    return reach[len(reach) // 2] if len(reach) >= min_n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', default='mg1')
    ap.add_argument('--state')
    a = ap.parse_args()
    steps, entropy = steps_of(a.run)
    steps = steps or 0
    try:
        cur = json.load(open(os.path.join(REPO, f'main_{a.run}_curriculum.json')))
    except FileNotFoundError:
        cur = {}
    eps = collections.defaultdict(list)
    try:
        for line in open(os.path.join(REPO, f'mg_log_{a.run}.jsonl')):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            eps[r['act']].append(r)
    except FileNotFoundError:
        pass
    st = {}
    if a.state and os.path.exists(a.state):
        st = json.load(open(a.state))
    now, last_look = time.time(), st.get('checked', 0)
    print(f'{a.run}: {steps:,} steps, {sum(map(len, eps.values())):,} episodes'
          + (f'; policy entropy {entropy:.2f}' if entropy is not None else ''))
    tail = sorted((r for rs in eps.values() for r in rs[-3000:]), key=lambda r: r['t'])[-3000:]
    steps_of_ep = lambda r: max(1, -(-(r['frames'] - sum(x.get('cost', 0) for x in r.get('stages', []))) // 4))
    tot = sum(map(steps_of_ep, tail)) or 1
    by_act = collections.Counter()
    for r in tail:
        by_act[r['act']] += steps_of_ep(r)
    print(f'  training time, last {len(tail):,} episodes: '
          + ', '.join(f'{a} {100 * v / tot:.0f}%' for a, v in sorted(by_act.items()))
          + f" (boss share {100 * sum(steps_of_ep(r) for r in tail if r.get('boss_share')) / tot:.0f}%,"
          f" ring share {100 * sum(steps_of_ep(r) for r in tail if r.get('ring_share')) / tot:.0f}%,"
          f" front share {100 * sum(steps_of_ep(r) for r in tail if r.get('front_share')) / tot:.0f}%)")
    alerts = []
    if entropy is not None and entropy > ENTROPY_MAX and not st.get('entropy_reported'):
        alerts.append(f'ALERT entropy: the policy is flattening ({entropy:.2f} > {ENTROPY_MAX}; '
                      f'mg2 collapsed this way when its imitation pull reached 0)')
        st['entropy_reported'] = True
    elif entropy is not None and entropy <= ENTROPY_MAX - 0.2:
        st['entropy_reported'] = False
    n_acts = max(1, len(cur))                                # stall counted per act's share of steps
    for key, c in sorted(cur.items()):
        act = key.split(':')[-1]
        rs = eps.get(act, [])
        bs = [r for r in rs if r.get('boss_share')]            # MG_BOSS_SHARE (from 2026-10-15)
        rg = [r for r in rs if r.get('ring_share')]             # MG_RING_SHARE (from 2026-10-15)
        fr = [r for r in rs if r.get('front_share')]            # MG_FRONT_SHARE (from 2026-10-15 ~15:30)
        last = [r for r in rs if not r.get('boss_share') and not r.get('ring_share')
                and not r.get('front_share')][-500:]           # comparable with before
        out = collections.Counter(r['outcome'] for r in last)
        fresh = [r for r in rs if r['start'] == 0 and not r.get('ring_share') and not r.get('front_share')]
        fc = [r for r in fresh if r['outcome'] == 'clear']
        recent_f = fresh[-100:]
        target = fresh[0]['target'] if fresh else None
        stage_all = sum(r['outcome'] == 'stage' for r in rs)
        played = [r for r in rs if r.get('stages')]          # MG_SS=pilot: a special stage played
        won = sum(st.get('emeralds', 0) for r in played for st in r['stages'])
        owed = [st for r in played for st in r['stages'] if 'bonus' in st]    # MG_EMERALD_PAY=return
        print(f"  {act}: start point {c['ptr']} ({c['frac']:.1%}); last {len(last)}: "
              + ', '.join(f'{k} {v}' for k, v in out.most_common()))
        if bs:
            bo = collections.Counter(r['outcome'] for r in bs[-500:])
            print(f"    boss share: {len(bs)} episodes; last {len(bs[-500:])}: "
                  + ', '.join(f'{k} {v}' for k, v in bo.most_common())
                  + f"; matched him {sum(r['success'] for r in bs[-500:])}")
        front = fresh_frontier(fresh)                            # coordinator.fresh_frontier's rule
        if front is not None and target:
            line = f"    fresh-start frontier: {front / target:.1%} into the act (median of the last 32)"
            if fr:
                past = [r for r in fr[-100:] if r['start_frac'] * target + r['progress'] * r['target'] > front + 60]
                line += (f"; front share {len(fr)} episodes, last {len(fr[-100:])}: got 1 s past it "
                         f"{len(past)}, " + ', '.join(f'{k} {v}' for k, v in
                                                 collections.Counter(r['outcome'] for r in fr[-100:]).most_common()))
            print(line)
        hurt = [h for r in last for h in r.get('hurt', [])]      # logged from 2026-10-15
        if hurt:
            lost = [h[1] - h[2] for h in hurt if h[2] < h[1]]
            print(f"    hits survived: {len(hurt)} in the last {len(last)} episodes, {len(lost)} lost rings"
                  + (f" (median {sorted(lost)[len(lost) // 2]})" if lost else ''))
        for ring in sorted({tuple(r['ring_share']) for r in rg}):
            tries = [r for r in rg if tuple(r['ring_share']) == ring]
            took = [r for r in tries if list(ring) in r.get('giant_rings', [])]
            back = [r for r in took if any('bonus_paid_at' in x for x in r.get('stages', []))]
            print(f"    ring share, giant ring at ({ring[2]},{ring[3]}): {len(tries)} tries, took it {len(took)}"
                  f" (last 100: {sum(list(ring) in r.get('giant_rings', []) for r in tries[-100:])}),"
                  f" got back after {len(back)}")
        drowned = sum(r.get('cause') == 'drown' for r in last)
        if drowned:
            print(f"    drowned: {drowned} of the last {len(last)} episodes")
        bh = [r['boss_hits'] for r in last if 'boss_hits' in r]   # MG_BOSS (from 2026-10-14)
        if any(bh):
            print(f"    boss hits: {sum(bh)} in the last {len(bh)} episodes ({sum(h > 0 for h in bh)} with any)")
        s = st.setdefault(key, {'ptr': c['ptr'], 'since': steps})
        if s['since'] > steps:                               # a rollback (mg2 2026-10-12) left it in the future
            s['since'] = steps
        if c['ptr'] != s['ptr']:                             # moved back: no start was at it before our last
            s.update(ptr=c['ptr'], since=steps, stall_reported=False,     # look; forward (stones renumber,
                     floor=last_look if c['ptr'] < s['ptr'] else now)     # a rollback): only from now
        at_all = since_arrival(rs, c['ptr'], s.get('floor', 0))          # began AT it since it moved here
        at = at_all[-32:]
        if at:                                               # what the at-pointer rule judges (mg1, mg2)
            print(f"    at the start point itself: {sum(r['success'] for r in at)} of the last {len(at)} matched him"
                  f" ({len(at_all)} started there)")
        works, rate, n = next_point_works(rs, c['ptr'], at_all)
        if len(at_all) >= STONES_AT and c['ptr'] > 0 and not works:     # the whole stretch is hard
            print(f"    stalled but hard throughout: start point {c['ptr'] + 1} matched {rate:.0%} of {n}"
                  f" (stones need the next one to work)")
        elif len(at_all) >= STONES_AT and c['ptr'] > 0 and s_get(st, key, 'stones_reported') != c['ptr']:
            alerts.append(f"ALERT stepping stones: {act} start point {c['ptr']}: {len(at_all)} starts there, "
                          f"{sum(r['success'] for r in at)} of the last {len(at)} matched")
            st.setdefault(key, {})['stones_reported'] = c['ptr']
        print(f"    fresh starts {len(fresh)}, clears {len(fc)}"
              + (f" (last 100: {sum(r['outcome'] == 'clear' for r in recent_f)}), best {min(r['frames'] for r in fc)}"
                 f" vs his {target}" if fc else '')
              + f"; giant-ring exits {stage_all} of {len(rs)} ({stage_all / max(1, len(rs)):.1%}),"
                f" last 500 {out.get('stage', 0)}"
              + (f"; special stages played in {len(played)} episodes (last 500: "
                 f"{sum(bool(r.get('stages')) for r in last)}), emeralds won {won}" if played else '')
              + (f"; got back after {sum('bonus_paid_at' in st for st in owed)} of the {len(owed)} "
                 f"paid on return" if owed else ''))
        if steps - s['since'] >= STALL_STEPS * n_acts and c['ptr'] > 0:
            line = f"stall: {act} start point {c['ptr']} unmoved for {steps - s['since']:,} steps"
            if s.get('stall_reported'):
                print('   ', line)                         # reported once per start point
            else:
                alerts.append('ALERT ' + line)
                s['stall_reported'] = True
        if act == 'AIZ1' and steps >= GATE_STEPS and c['ptr'] >= GATE_PTR and not st.get('gate_reported'):
            alerts.append(f"ALERT wall gate: {steps:,} steps and AIZ1 still at start point {c['ptr']}")
            st['gate_reported'] = True
        if act == 'AIZ1' and c['ptr'] < GATE_PTR and not st.get('gate_passed'):
            st['gate_passed'] = steps
            print(f'    NEWS: past the wall (start point {c["ptr"]} < {GATE_PTR}) at {steps:,} steps')
        if fc and not st.get(f'first_fresh_{act}'):
            st[f'first_fresh_{act}'] = steps
            print(f"    NEWS: first fresh clear of {act} ({fc[0]['frames']} frames)")
    for x in alerts:
        print(x)
    st['checked'] = now
    if a.state:
        with open(a.state + '.tmp', 'w') as f:
            json.dump(st, f, indent=1)
        os.replace(a.state + '.tmp', a.state)


if __name__ == '__main__':
    main()
