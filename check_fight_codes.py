#!/usr/bin/env python3
"""check_fight_codes.py — does MG_FIGHT_CODES turn MGZ1's unflagged end boss into a fight, and change nothing else?

    python3 check_fight_codes.py [RUN]          # default mg2

1. the run's acts built without and with MG_FIGHT_CODES: MGZ1 gains one fight, from the boss object's
   appearance (~t 13,068 in his run) to the clear, with its object code ($00088572) as the act's fight code;
   every other act's fights, fight codes and start points' boss flags are unchanged
2. MGZ1's start points: boss flag on inside that fight, off everywhere else (incl. ~t 2,957, where the
   drill boss hovers over the level as a prop)
3. at run time, his inputs from MGZ1's first start point in the fight: the env is in the fight (route.fight
   set) from the first step until the kill, `stuck` never fires though his longest gap between hits is ~420
   frames, and the clear arrives with route progress at the act's end
4. his inputs from a start point at the prop (~t 2,957) for 600 frames: never in a fight
CPU only. Ends with ALL CHECKS PASS or the failures.
"""
import os, sys
HOME = os.path.dirname(os.path.abspath(__file__))
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
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='', MG_SS='truncate', MG_FIGHT_CODES='0')
    import coordinator as C
    import demos
    if not C.MG_BOSS:
        sys.exit('needs the boss tracker (RUNS boss_bar or boss_pay)')
    acts = train_main.RUNS[run]['acts']

    def build(flag):
        C.MG_FIGHT_CODES = flag
        env = C.SonicCoordinator(acts=acts, demo_prob=0.0)
        env.demo_info()
        return env

    print('1. built without and with MG_FIGHT_CODES')
    a = build(False)
    before = {s['name']: (list(s['route']['fights']), [c['boss'] for c in s['ckpts']]) for s in a._segs}
    a.close()
    env = build(True)
    for s in env._segs:
        f0, b0 = before[s['name']]
        f1, b1 = list(s['route']['fights']), [c['boss'] for c in s['ckpts']]
        if s['name'] == 'MGZ1':
            new = [f for f in f1 if f not in f0]
            ok = (len(new) == 1 and f0 == [] and 12900 <= new[0][0] <= 13200 and new[0][1] == s['T']
                  and s.get('fight_codes') == {0x00088572})
            check(ok, f"MGZ1: fights {f0} -> {f1}, codes {[hex(c) for c in s.get('fight_codes', ())]}")
            mg = s
        else:
            check(f1 == f0 and b1 == b0 and not s.get('fight_codes'),
                  f"{s['name']}: fights and start-point boss flags unchanged ({len(f1)} fights)")

    print('\n2. MGZ1 start points')
    lo, hi = [f for f in mg['route']['fights']][0]
    on = [k for k, c in enumerate(mg['ckpts']) if c['boss']]
    want = [k for k, c in enumerate(mg['ckpts']) if lo <= c['t'] < hi]
    check(on == want and on, f'boss flag on at start points {on[0] if on else "-"}..{on[-1] if on else "-"} (t {lo}..{hi}), off elsewhere')
    prop = min(range(len(mg['ckpts'])), key=lambda k: abs(mg['ckpts'][k]['t'] - 2957))
    check(not mg['ckpts'][prop]['boss'], f"start point {prop} (t {mg['ckpts'][prop]['t']}, the hovering drill): no fight")

    path = next(p for p in env.demo_files if demos.demo_id(p) == mg['demo'])
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    orig, at = C.emu_frame, {'f': 0}

    def his(e, _mask):
        at['f'] += 1
        return orig(e, table[actions[at['f']]])

    print('\n3. his inputs through the fight')
    k0 = on[0]
    C.emu_frame = his
    try:
        env.reset(options={'key': mg['key'], 'ckpt': k0})
        at['f'] = mg['ckpts'][k0]['frame']
        in_fight, done, steps, res = [], False, 0, None
        while not done:
            _, _, term, trunc, info = env.step(0)
            done, steps = term or trunc, steps + 1
            in_fight.append(env._route.fight is not None)
        res = info['mg_result']
        check(res['outcome'] == 'clear' and all(in_fight[:-1]),
              f"{res['outcome']} after {res['frames']} frames; in the fight on {sum(in_fight)} of {len(in_fight)} steps; "
              f"progress {res['progress']:.0%}")

        print('\n4. his inputs at the prop')
        env.reset(options={'key': mg['key'], 'ckpt': prop})
        at['f'] = mg['ckpts'][prop]['frame']
        fights = 0
        for _ in range(150):
            _, _, term, trunc, info = env.step(0)
            fights += env._route.fight is not None
            if term or trunc:
                break
        check(fights == 0, f'{fights} steps in a fight over ~600 frames from t {mg["ckpts"][prop]["t"]}')
    finally:
        C.emu_frame = orig
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
