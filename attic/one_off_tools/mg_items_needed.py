#!/usr/bin/env python3
"""mg_items_needed.py — did Tyler's stars and fire shields carry him? Replays his inputs through each
item stretch found by mg_hazards.py with the item taken away, and with it kept (the control).

    python3 attic/one_off_tools/mg_items_needed.py [demos/play_<stamp>.npz] [--items invincible,fire shield]

Per stretch: from the first start point a little after he got the item, clear its bit in Player 1's
status_secondary ($FFB02B; bit 1 invincible, 0+4 fire shield) and play HIS inputs through the env until
the stretch ends (+1 s). Counts hits (rings knocked to 0, or a shield lost) and whether he dies. With
the item kept the same replay must be clean (as he played it). A stretch where the item-less replay
dies or gets hit is one where the agent, once its start point moves back past the monitor, has to
break that monitor itself, or find another way through. CPU only, one core.
"""
import json
import os
import sys

REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MG_ENV_VERSION', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MG_LOG'] = os.devnull
import numpy as np                      # noqa: E402

import coordinator as C                 # noqa: E402
import demos                            # noqa: E402

STATUS2 = 0xFFB02B
CLEAR = {'invincible': 0x02, 'fire shield': 0x11}     # bits to clear (fire: bit 4 + the any-shield bit 0)
X, RINGS = 0xB010, 0xFE20


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else 'demos/play_20261010-202923.npz'
    kinds = set(CLEAR)
    if '--items' in sys.argv:
        kinds = set(sys.argv[sys.argv.index('--items') + 1].split(','))
    stamp = os.path.basename(path).replace('play_', '').replace('.npz', '')
    hz = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mg_hazards_{stamp}.json')))
    todo = [(act, it) for act, r in hz.items() for it in r['items'] if it['item'] in kinds and it['frames'] >= 300]
    acts = sorted({a for a, _ in todo})
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    env = C.SonicCoordinator(demo_files=[path], acts=acts)
    env.demo_info()
    segs = {s['name']: s for s in env._segs}
    orig = C.emu_frame
    at = {'f': 0, 'swap': None}
    rec = {'rings': [], 's2': []}

    def his_input(e, _mask):
        at['f'] += 1
        out = orig(e, table[actions[at['f']]])
        raw = np.frombuffer(bytes(e.get_ram()), dtype=np.uint8)
        if at['swap'] is None:
            x = int(out[1]['x'])
            if x >> 8 != x & 0xFF:
                at['swap'] = (int(raw[X]) << 8 | int(raw[X + 1])) != x
        ram = raw.reshape(-1, 2)[:, ::-1].reshape(-1) if at['swap'] else raw
        rec['rings'].append(int(ram[RINGS]) << 8 | int(ram[RINGS + 1]))
        rec['s2'].append(int(ram[STATUS2 & 0xFFFF]))
        return out

    def run(seg, k, end_abs, take_away):
        for v in rec.values():
            v.clear()
        env.reset(options={'key': seg['key'], 'ckpt': k})
        at['f'] = seg['ckpts'][k]['frame']
        if take_away:
            d = env.base_env.data
            d.set_variable('probe_status2', {'address': STATUS2, 'type': '|u1'})
            d.set_value('probe_status2', int(d.lookup_value('probe_status2')) & ~CLEAR[take_away] & 0xFF)
        outcome, done = 'reached the end of the stretch', False
        while not done and at['f'] < end_abs:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
            if done:
                outcome = info['mg_result']['outcome']
        r = rec['rings']
        ring_hits = sum(1 for i in range(1, len(r)) if r[i] == 0 and r[i - 1] > 0)
        return outcome, ring_hits

    C.emu_frame = his_input
    rows = []
    try:
        for act, it in todo:
            seg = segs[act]
            base = seg['ckpts'][0]['frame'] + 1                 # mg_hazards' frame 0
            frames = [c['frame'] for c in seg['ckpts']]
            k = int(np.searchsorted(frames, base + it['t0'] + 30))
            if k >= len(frames) or frames[k] > base + it['t1'] - 240:
                print(f"{act} {it['item']} {it['t0']}-{it['t1']}: no start point inside the stretch", flush=True)
                continue
            end_abs = base + it['t1'] + 60
            kept = run(seg, k, end_abs, None)
            gone = run(seg, k, end_abs, it['item'])
            row = dict(act=act, item=it['item'], t0=it['t0'], t1=it['t1'], start_point=k,
                       from_frame=frames[k] - base, kept=kept, taken_away=gone)
            rows.append(row)
            print(f"{act:5s} {it['item']:12s} frames {it['t0']}-{it['t1']} (from start point {k}, frame {frames[k] - base}): "
                  f"kept -> {kept[0]}, {kept[1]} ring hits;  taken away -> {gone[0]}, {gone[1]} ring hits", flush=True)
    finally:
        C.emu_frame = orig
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mg_items_needed_{stamp}.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
