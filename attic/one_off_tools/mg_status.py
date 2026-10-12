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

REPO = '/home/tyler/Projects/sonic-rl'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', default='mg1')
    ap.add_argument('--state')
    a = ap.parse_args()
    steps, entropy = steps_of(a.run)
    steps = steps or 0
    try:
        cpath = os.path.join(REPO, f'main_{a.run}_curriculum.json')
        cur, since = json.load(open(cpath)), os.path.getmtime(cpath)
    except FileNotFoundError:
        cur, since = {}, 0
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
    print(f'{a.run}: {steps:,} steps, {sum(map(len, eps.values())):,} episodes'
          + (f'; policy entropy {entropy:.2f}' if entropy is not None else ''))
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
        last = rs[-500:]
        out = collections.Counter(r['outcome'] for r in last)
        fresh = [r for r in rs if r['start'] == 0]
        fc = [r for r in fresh if r['outcome'] == 'clear']
        recent_f = fresh[-100:]
        target = fresh[0]['target'] if fresh else None
        stage_all = sum(r['outcome'] == 'stage' for r in rs)
        played = [r for r in rs if r.get('stages')]          # MG_SS=pilot: a special stage played
        won = sum(st.get('emeralds', 0) for r in played for st in r['stages'])
        print(f"  {act}: start point {c['ptr']} ({c['frac']:.1%}); last {len(last)}: "
              + ', '.join(f'{k} {v}' for k, v in out.most_common()))
        drowned = sum(r.get('cause') == 'drown' for r in last)
        if drowned:
            print(f"    drowned: {drowned} of the last {len(last)} episodes")
        at_all = [r for r in rs if r['start'] == c['ptr'] and r['t'] >= since]   # since it moved here
        at = at_all[-32:]                                    # (the trainer rewrites the file on each move)
        if at:                                               # what the at-pointer rule judges (mg1, mg2)
            print(f"    at the start point itself: {sum(r['success'] for r in at)} of the last {len(at)} matched him"
                  f" ({len(at_all)} started there)")
        if len(at_all) >= STONES_AT and c['ptr'] > 0 and s_get(st, key, 'stones_reported') != c['ptr']:
            alerts.append(f"ALERT stepping stones: {act} start point {c['ptr']}: {len(at_all)} starts there, "
                          f"{sum(r['success'] for r in at)} of the last {len(at)} matched")
            st.setdefault(key, {})['stones_reported'] = c['ptr']
        print(f"    fresh starts {len(fresh)}, clears {len(fc)}"
              + (f" (last 100: {sum(r['outcome'] == 'clear' for r in recent_f)}), best {min(r['frames'] for r in fc)}"
                 f" vs his {target}" if fc else '')
              + f"; giant-ring exits {stage_all} of {len(rs)} ({stage_all / max(1, len(rs)):.1%}),"
                f" last 500 {out.get('stage', 0)}"
              + (f"; special stages played in {len(played)} episodes (last 500: "
                 f"{sum(bool(r.get('stages')) for r in last)}), emeralds won {won}" if played else ''))
        s = st.setdefault(key, {'ptr': c['ptr'], 'since': steps})
        if c['ptr'] != s['ptr']:
            s.update(ptr=c['ptr'], since=steps, stall_reported=False)
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
    if a.state:
        with open(a.state + '.tmp', 'w') as f:
            json.dump(st, f, indent=1)
        os.replace(a.state + '.tmp', a.state)


if __name__ == '__main__':
    main()
