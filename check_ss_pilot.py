#!/usr/bin/env python3
"""check_ss_pilot.py — does the main game hand special stages to the special-
stage agent correctly? (ss_pilot.py, coordinator.py MG_SS=pilot)

    python3 check_ss_pilot.py [model.zip]     # default: the d9 run's latest save

CPU only, ~4 min. Needs the giant-ring fixtures in states/handoff/ (the private
states repo): aiz1_ring_minus1 (AIZ1's ring, 1 frame before it; the game's
first special stage = board 1) and mhz1_ring7_minus1 (MHZ1's ring with all 7
Chaos Emeralds -> Hidden Palace). Checks:
  1. Tyler's stage-1 inputs, played from the main game's handover, clear
     102/102 in exactly his 2,925 frames (the stage is the one he recorded on)
  2. the agent (deterministic) sees identical observations and makes identical
     choices from the handover as from the captured curriculum_stage_1.state,
     over its whole run
  3. a main-game episode through AIZ1's ring: the stage is played, the game
     puts Sonic back at the ring (7112, 1216) with the act clock running, the
     episode goes on, and the stage is in its log line; refund (default) charges
     every frame and pays it all back for the emerald, real charges it all, game
     only the reload; the emerald bonus pays on top
  4. Hidden Palace: with gray Super Emeralds, the episode's step walks to the
     preferred emerald (boards 8, then 11, then the automatic choice), Tyler's
     inputs for that board clear it, the Super Emerald is won, and the game
     returns to MHZ1 at the ring
  5. MG_SS unset: the ring still ends the episode as 'stage' (the default)
Ends with ALL CHECKS PASS or the failures.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HOME, 'states', 'handoff')
failures = []


def check(ok, what):
    print(f"  {'ok ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        failures.append(what)


def child(mode, model, extra_env):
    """One check in its own process (coordinator reads MG_* at import)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MG_ENV_VERSION='2', **extra_env)
    out = subprocess.run([sys.executable, __file__, '--child', mode, model], env=env,
                         capture_output=True, text=True, cwd=HOME)
    lines = [l for l in out.stdout.splitlines() if l.startswith('{')]
    if out.returncode or not lines:
        print(out.stdout[-2000:], out.stderr[-3000:])
        return None
    return json.loads(lines[-1])


def run_child(mode, model):
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import coordinator as C
    import demos
    import ss_pilot
    ring = open(os.path.join(FIX, 'aiz1_ring_minus1.state'), 'rb').read()
    ring7 = open(os.path.join(FIX, 'mhz1_ring7_minus1.state'), 'rb').read()

    def his(board):
        return demos.load_demo(glob.glob(os.path.join(HOME, 'demos', f'stage{board}_*.npz'))[0])[0]

    if mode in ('inputs', 'same'):
        emu = C.make_retro()
        pilot = ss_pilot.SSPilot(emu, run='d9', model=model, deterministic=True)

        def enter():
            emu.reset()
            emu.em.set_state(ring)
            C.emu_frame(emu, C.ACTION_BUTTONS[0])
        if mode == 'inputs':
            enter()
            res = pilot.play(script=his(1))
            print(json.dumps(dict(res, steps=pilot.env.steps, his=len(his(1)))))
            return
        env, ta = pilot.env, []
        obs, _ = env.reset(options={'stage_file': pilot.sse.stage_file(1), 'stage': 1})
        lstm, start = None, True
        while True:
            a, lstm = pilot.model.predict(obs, state=lstm, episode_start=np.array([start]),
                                          deterministic=True)
            start = False
            a = int(np.asarray(a).reshape(-1)[0])
            ta.append(({k: v.copy() for k, v in obs.items()}, a))
            obs, _, done, trunc, _ = env.step(a)
            if done or trunc:
                break
        enter()
        tb = []
        pilot.play(trace=tb)
        same = len(ta) == len(tb) and all(
            aa == ab and all(np.array_equal(oa[k], ob[k]) for k in oa)
            for (oa, aa), (ob, ab) in zip(ta, tb))
        print(json.dumps(dict(decisions=[len(ta), len(tb)], same=bool(same))))
        return

    # main-game episodes (AIZ1's segments are quick to build; the rings' states
    # are loaded over the episode's start point)
    env = C.SonicCoordinator(demo_files=[os.path.join(HOME, 'demos', 'play_20261010-202923.npz')],
                             acts=['AIZ1'], demo_prob=1.0)
    env.demo_info()
    env.reset(options={'key': env._segs[0]['key'], 'ckpt': 8})
    emu = env.base_env
    board = None
    if mode.startswith('super'):
        board = int(mode[5:]) if mode[5:].isdigit() else None
        emu.em.set_state(ring7)
        for i in range(7):
            emu.data.set_variable(f'em_arr{i}', {'address': 0xFFFFB2 + i, 'type': '|u1'})
            emu.data.set_value(f'em_arr{i}', 2)             # gray on the pedestals
    else:
        emu.em.set_state(ring)
        board = 1
    if mode != 'default':
        env._pilot = ss_pilot.SSPilot(emu, run='d9', model=model)
        play, shrine, chosen = env._pilot.play, env._pilot.shrine, {}

        def shrine_and_note(prefer, buttons):
            got = shrine(prefer, buttons)
            chosen['board'] = got[2]
            return got
        env._pilot.shrine = shrine_and_note
        env._pilot.play = lambda: play(script=his(chosen.get('board') or board))
    f0 = env.frames
    _, r, term, trunc, info = env.step(0)
    out = dict(reward=r / C.REWARD_SCALE, frames=env.frames - f0, term=term, trunc=trunc,
               zone=int(info['zone']), x=int(info['x']), y=int(info['y']),
               timer_on=int(info['mg_hud_timer']), stages=env._stages,
               outcome=info.get('mg_result', {}).get('outcome'))
    if not (term or trunc):
        n = 0
        while not (term or trunc):
            _, r, term, trunc, info = env.step(2)
            n += 1
        out['then'] = info['mg_result']['outcome']
        out['logged'] = json.loads(open(os.environ['MG_LOG']).read().splitlines()[-1]).get('stages')
    print(json.dumps(out))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, 'sonic_ss_multistage_d9.zip')
    if not os.path.exists(os.path.join(FIX, 'aiz1_ring_minus1.state')):
        sys.exit(f'missing fixtures in {FIX}')
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, 'model.zip')
        subprocess.run(['cp', model, snap], check=True)    # a running trainer rewrites its save
        log = os.path.join(tmp, 'mg.jsonl')

        print('1. Tyler\'s stage-1 inputs from the main game\'s handover')
        a = child('inputs', snap, {})
        check(a and a['outcome'] == 'clear' and a['spheres'] == 102 and a['steps'] == a['his'] == 2925
              and a['board'] == 1, f"board {a and a['board']}, {a and a['outcome']} {a and a['spheres']}/102 "
              f"in {a and a['steps']} frames (his 2925); handover {a and a['to_run']} frames after the ring")
        print('2. the agent\'s view: handover vs the captured state')
        b = child('same', snap, {})
        check(b and b['same'], f"identical observations and choices: {b and b['same']} "
              f"({b and b['decisions']} decisions)")
        print('3. a main-game episode through AIZ1\'s giant ring')
        for t in ('refund', 'real', 'game'):
            c = child('chaos', snap, dict(MG_SS='pilot', MG_SS_TIME=t, MG_EMERALD_BONUS='300', MG_LOG=log))
            st = (c or {}).get('stages') or [{}]
            s0 = st[0]
            want_cost = s0.get('reload', -1) + (0 if t == 'game' else s0.get('in_stage', 0))
            want_paid = 300 + (want_cost if t == 'refund' else 0)
            check(c and not c['term'] and not c['trunc'] and (c['x'], c['y']) == (7112, 1216)
                  and c['timer_on'] and s0.get('outcome') == 'clear' and s0.get('emeralds') == 1
                  and s0.get('cost') == want_cost and s0.get('paid') == want_paid
                  and c['frames'] == 1 + want_cost
                  and abs(c['reward'] - (want_paid - want_cost)) < 1e-6 and c.get('logged') == st,
                  f"{t} time: {s0.get('outcome')}, emeralds +{s0.get('emeralds')}, back at "
                  f"({c and c['x']},{c and c['y']}) clock on {c and c['timer_on']}, cost {s0.get('cost')} "
                  f"(reload {s0.get('reload')}, in stage {s0.get('in_stage')}), paid {s0.get('paid')}, "
                  f"reward {c and round(c['reward'])}, "
                  f"episode went on to '{c and c.get('then')}' and logged the stage")
        print('4. Hidden Palace (all 7 Chaos Emeralds, gray Super Emeralds)')
        for pref in ('8', '11', 'auto'):
            d = child(f'super{pref}', snap, dict(
                MG_SS='pilot', MG_SS_SUPER=pref if pref == 'auto' else f'{pref},9,10,12,13,14', MG_LOG=log))
            s0 = ((d or {}).get('stages') or [{}])[0]
            check(d and not d['trunc'] and d['zone'] == 7 and (d['x'], d['y']) == (8768, 1088)
                  and s0.get('board') == s0.get('expected_board')
                  and (pref == 'auto' or s0.get('board') == int(pref))
                  and s0.get('outcome') == 'clear' and s0.get('emeralds') == 1,
                  f"prefer board {pref}: touched slot {s0.get('slot')} after {s0.get('shrine')} frames, "
                  f"board {s0.get('board')} {s0.get('outcome')} {s0.get('spheres')}/{s0.get('total')}, "
                  f"emeralds +{s0.get('emeralds')}, back in zone {d and d['zone']} at ({d and d['x']},{d and d['y']})")
        print('5. MG_SS unset (the default)')
        e = child('default', snap, dict(MG_LOG=log))
        check(e and e['trunc'] and e['outcome'] == 'stage' and not e['stages'],
              f"the ring ends the episode as '{e and e['outcome']}'")
    print('\nALL CHECKS PASS' if not failures else f'\n{len(failures)} FAILED:\n  ' + '\n  '.join(failures))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        os.environ.setdefault('MG_LOG', os.devnull)
        sys.path.insert(0, HOME)
        run_child(sys.argv[2], sys.argv[3])
    else:
        main()
