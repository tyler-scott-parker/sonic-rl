#!/usr/bin/env python3
"""check_hurt_log.py — does the episode log record the hits Sonic survives ('hurt': [frame, rings
held, rings left] per hit), the first step of the ring value (Tyler 2026-10-15, "option 2")?

    python3 check_hurt_log.py [RUN]          # default mg2

Random inputs from the run's boss-share starts (inside his fights, where hits are common) until a
few hits are logged, watching every frame through the env's own emu_frame:
1. each logged hit is a frame where the player routine turned to hurt (4) in the level
2. each frame that turned to hurt is logged, once (none missed, none doubled)
3. rings held = the count the frame before; rings left = the count on the hit frame, which is 0
   unless a shield took the hit (then unchanged)
4. a death is never logged as a hit
CPU only; the episode log goes to a temp file. Ends with ALL CHECKS PASS or the failures.
"""
import json
import os
import random
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
WANT_HITS, MAX_EPISODES = 6, 40
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import coordinator as C
    log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    C.LOG_FILE = log
    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    env.set_demo_pointers({k: v['ptr'] for k, v in cur.items()})
    C.MG_BOSS_SHARE = 1.0 if env.boss_ranges() else 0.0

    seen = []                                       # per episode: [(frame, rings before, rings now, died)]
    orig = C.emu_frame

    def watch(e, mask):
        raw, info = orig(e, mask)
        w = watch.state
        w['f'] += 1
        rt, rings = int(info['mg_routine']), int(info['rings'])
        in_level = C.frame_flags(info)[0]
        if rt == C.HURT_ROUTINE and w['rt'] != C.HURT_ROUTINE and in_level:
            w['hits'].append([w['f'], w['rings'], rings])
        if rt in C.DEAD_ROUTINES and w['rt'] not in C.DEAD_ROUTINES:
            w['deaths'].append(w['f'])
        w['rt'], w['rings'] = rt, rings
        return raw, info
    C.emu_frame = watch

    random.seed(3)
    total = 0
    for ep in range(MAX_EPISODES):
        env.reset()
        watch.state = dict(f=0, rt=None, rings=0, hits=[], deaths=[])
        done = False
        while not done:
            _, _, term, trunc, info = env.step(random.randrange(len(C.ACTION_BUTTONS)))
            done = term or trunc
        if env._stages:                              # frames inside a special stage aren't the env's
            continue
        line = json.loads(open(log).read().strip().splitlines()[-1])
        seen.append((line, watch.state))
        total += len(watch.state['hits'])
        if total >= WANT_HITS:
            break
    C.emu_frame = orig
    os.unlink(log)

    hits = [(line, h) for line, w in seen for h in w['hits']]
    print(f'{len(seen)} episodes ({", ".join(sorted({l["act"] for l, _ in seen}))}), {len(hits)} hits survived')
    check(len(hits) >= WANT_HITS, f'at least {WANT_HITS} hits to judge')
    for line, w in seen:
        logged = line.get('hurt', [])
        check(logged == w['hits'], f"{line['act']} start {line['start']} ({line['outcome']}): logged {logged} "
                                   f"= watched {w['hits']}")
        check(not any(f in w['deaths'] for f, _, _ in logged), '  no death frame among them')
        for f, held, left in logged:
            check(left == 0 if held > 0 and left != held else True,
                  f'  frame {f}: {held} rings held, {left} left'
                  + (' (shield took it)' if held > 0 and left == held else ''))
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
