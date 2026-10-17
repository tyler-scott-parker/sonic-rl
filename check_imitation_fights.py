#!/usr/bin/env python3
"""check_imitation_fights.py — with RUNS imitation 'fights', does the imitation pull's focused half cover his
whole boss fight while an act's pointer sits inside it, and change nothing else?

    python3 check_imitation_fights.py [RUN]          # default mg2

1. demo_info() carries each act's fights, the same windows the boss share uses (route['fights']), and
   MGZ2's end fight spans the start points it should (his Boss_flag, ~268..318 on 2026-10-17)
2. 'fights' with no fight windows known = the focused frames unchanged
3. with them: the focused frames = before + his frames in each fight that holds an act's pointer, nothing else
4. a pointer moved into another act's fight (temp curriculum) brings that fight in; one outside any fight
   brings nothing
5. what the fight frames teach (his actions there) and each act's share of the focused half, before/after
CPU only, no model; builds the run's acts in-process (start_cache: ~1-2 min). Ends with ALL CHECKS PASS or
the failures.
"""
import collections
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
    sys.path.insert(0, HOME)
    import numpy as np
    import train_main as T
    T.configure_env(run)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import coordinator as C
    cfg = {**T.IMITATION_DEFAULTS, **T.RUNS[run]['imitation']}
    check(bool(cfg.get('fights')), f"RUNS {run} imitation fights={cfg.get('fights')}")
    cur_path = T.paths(run)['curriculum']
    cur = json.load(open(cur_path))

    print('1. demo_info() fights')
    env = C.SonicCoordinator(acts=T.RUNS[run]['acts'], demo_prob=0.0)
    infos = env.demo_info()
    fights = {d['key']: d['fights'] for d in infos}
    segs = {s['key']: s for s in env._segs}
    check(all(fights[k] == [tuple(f) for f in segs[k]['route']['fights']] for k in fights),
          "each act's fights = route['fights'] (what boss_ranges uses)")
    for d in infos:
        ck = segs[d['key']]['ckpts']
        spans = []
        for a, b in d['fights']:
            ks = [k for k, c in enumerate(ck) if a <= c['t'] < b]
            spans.append(f"t {a}..{b} = start points {ks[0]}..{ks[-1]}" if ks else f"t {a}..{b} (no start point)")
        print(f"    {d['name']}: " + ('; '.join(spans) if spans else 'none'))
    mg = next((d for d in infos if d['name'] == 'MGZ2'), None)
    if mg:
        ck = segs[mg['key']]['ckpts']
        end = [(a, b) for a, b in mg['fights'] if a <= ck[270]['t'] < b]
        check(len(end) == 1, f"MGZ2 start point 270 (t {ck[270]['t']}) lies in one fight: {end}")

    cb = T.ImitationCallback(dict(cfg, fights=True), run, cur_path)
    cb.load_data()
    S = cb.S

    print('\n2. no fight windows known: unchanged')
    before = set(T.ImitationCallback.frontier(cb).tolist())
    cb_off = T.ImitationCallback(dict(cfg, fights=False), run, cur_path)
    cb_off.load_data()
    check(before == set(cb_off.frontier().tolist()), f'{len(before):,} focused frames either way')

    print('\n3. with the fights: + the fights holding a pointer, nothing else')
    cb._fights = fights
    after = set(cb.frontier().tolist())
    extra, held = set(), []
    for key, idx in cb.keys.items():
        t0 = cur[key]['frac'] * int(S['T'][idx[0]])
        t = S['t'][idx]
        for a, b in fights.get(key, ()):
            if a <= t0 < b:
                w = idx[(t >= a) & (t <= b)]
                extra |= set(w.tolist())
                held.append((key.split(':')[-1], a, b, t0, w))
    check(after == before | extra, f"{len(after):,} frames: + {len(extra - before):,} from the fights of "
          f"{', '.join(h[0] for h in held) or 'no act'}")
    check(any(h[0] == 'MGZ2' for h in held), 'MGZ2 (pointer in its end fight) is among them')

    print('\n4. a pointer moved into / out of a fight (temp curriculum)')
    other = next(((k, f) for k, f in fights.items() if f and k in cur and not any(h[0] == k.split(':')[-1] for h in held)), None)
    if other:
        key, (a, b) = other[0], other[1][0]
        Tk = int(S['T'][cb.keys[key][0]])
        fake = json.loads(json.dumps(cur))
        fake[key]['frac'] = (a + (b - a) / 2) / Tk                   # mid-fight
        with tempfile.NamedTemporaryFile('w', suffix='_curriculum.json', delete=False) as f:
            json.dump(fake, f)
        cb2 = T.ImitationCallback(dict(cfg, fights=True), run, f.name)
        cb2.load_data()
        cb2._fights = fights
        got = set(cb2.frontier().tolist())
        idx, t = cb.keys[key], S['t'][cb.keys[key]]
        want_new = set(idx[(t >= a) & (t <= b)].tolist())
        check(want_new <= got, f"{key.split(':')[-1]} pointer moved mid-fight (t {a}..{b}): its whole fight focused")
        outside = next(x for x in range(600, Tk - 600, 60)                # >= 800 frames from its every fight
                       if all(x < fa - 800 or x > fb + 800 for fa, fb in other[1]))
        fake[key]['frac'] = outside / Tk
        with open(f.name, 'w') as g:
            json.dump(fake, g)
        cb3 = T.ImitationCallback(dict(cfg, fights=False), run, f.name)
        cb3.load_data()
        on = set(cb2.frontier().tolist()) & set(idx.tolist())
        off = set(cb3.frontier().tolist()) & set(idx.tolist())
        check(on == off, f"{key.split(':')[-1]} pointer outside its fights (t {outside}): its frames the same "
                         f"with and without 'fights' ({len(on):,})")
        os.unlink(f.name)
    else:
        print('    (no other act with a fight to move into)')

    print('\n5. what the fight frames teach, and the focused half by act')
    names = C.ACTION_NAMES
    for act, a, b, t0, w in held:
        n = collections.Counter(int(x) for x in S['action'][w])
        tot = sum(n.values()) or 1
        print(f"    {act}: fight t {a}..{b}, pointer t {t0:.0f}; his {len(w):,} frames: "
              + ', '.join(f'{names[k]} {v / tot:.0%}' for k, v in n.most_common(6)))
        late = w[S['t'][w] >= t0 + 640]
        if len(late):
            n2 = collections.Counter(int(x) for x in S['action'][late])
            print(f"      of which past pointer + 640 (MGZ2: the Tails-carry flight), {len(late):,}: "
                  + ', '.join(f'{names[k]} {v / len(late):.0%}' for k, v in n2.most_common(6)))
    by_act = lambda sel: collections.Counter(int(S['act'][i]) for i in sel)
    acts = json.load(open(os.path.join(HOME, cfg['data'], 'meta.json')))['acts']
    b0, a0 = by_act(before), by_act(after)
    print('    share of the focused half, before -> after: ' + ', '.join(
        f'{acts[k]} {b0[k] / len(before):.0%} -> {a0[k] / len(after):.0%}' for k in sorted(set(b0) | set(a0))))

    print('\n' + ('ALL CHECKS PASS' if not bad else f'{len(bad)} FAILED: ' + '; '.join(bad)))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
