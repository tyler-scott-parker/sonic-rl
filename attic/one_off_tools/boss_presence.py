"""Where each boss object (by object code) is present in Tyler's recordings, touchable or not, and hit
or not: do the unflagged bosses also appear as props, and could their object code mark a fight? CPU only."""
import os, sys, glob, collections
REPO = os.path.expanduser('~/Projects/sonic-rl')
sys.path.insert(0, REPO)
os.chdir(REPO)
import train_main
train_main.configure_env('mg2')
os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
import numpy as np
import coordinator as C
import demos
CODES = {0x00088572: 'MGZ1 end boss (unflagged)', 0x0007764E: 'SOZ2 end boss (unflagged)',
         0x0007B288: 'SSZ1 last boss (unflagged)', 0x00080542: 'DEZ3 boss 2 (unflagged)',
         0x00080D30: 'DEZ3 boss 1 (unflagged, count 3)',
         0x00068A24: 'AIZ1 end boss', 0x0006923E: 'AIZ2 end boss', 0x00069F7A: 'HCZ1 end boss',
         0x0006AF0C: 'HCZ2 end boss', 0x0006C31A: 'MGZ2 end boss', 0x000684EC: 'AIZ1 fire-scene object'}
files = [p for p in os.environ['MG_DEMOS'].split(',') if p.strip()]
env = C.make_retro()
for path in files:
    actions, meta = demos.load_play(path)
    table = meta['action_table']
    env.reset()
    env.em.set_state(demos.play_start_state(meta))
    split = C.ActSplitter()
    open_iv = {}                                   # (code, slot) -> dict
    done = []
    print(f"\n=== {demos.demo_id(path)}", flush=True)
    for i, a in enumerate(actions):
        _, info = C.emu_frame(env, table[a])
        split.feed(i, info)
        cur = C.object_table(env, info)
        if cur is None:
            continue
        codes = (cur[:, 0].astype(np.int64) << 24 | cur[:, 1].astype(np.int64) << 16
                 | cur[:, 2].astype(np.int64) << 8 | cur[:, 3].astype(np.int64))
        name = split.cur['name'] if split.cur else '-'
        t = split.cur['t'] if split.cur else None
        seen = set()
        for sl in np.flatnonzero(np.isin(codes, list(CODES))):
            key = (int(codes[sl]), int(sl))
            seen.add(key)
            row = cur[sl]
            iv = open_iv.get(key)
            if iv is None:
                iv = open_iv[key] = dict(code=key[0], act=name, t0=t, f0=i, touch=0, counts=set(), boss_flag=0, n=0,
                                         x=int(row[0x10]) << 8 | int(row[0x11]), y=int(row[0x14]) << 8 | int(row[0x15]))
            iv['t1'], iv['n'] = t, iv['n'] + 1
            iv['touch'] += int(row[0x28] != 0)
            iv['counts'].add(int(row[0x29]))
            iv['boss_flag'] += int(info['mg_boss'] != 0)
        for key in [k for k in open_iv if k not in seen]:
            done.append(open_iv.pop(key))
    done += open_iv.values()
    merged = collections.defaultdict(list)
    for iv in sorted(done, key=lambda d: d['f0']):
        merged[iv['code']].append(iv)
    for code, ivs in merged.items():
        print(f"{CODES[code]} ${code:08X}:")
        for iv in ivs:
            if iv['n'] < 3:
                continue
            cs = sorted(iv['counts'])
            print(f"   {iv['act']} t {iv['t0']}..{iv['t1']} ({iv['n']} frames) at ({iv['x']},{iv['y']}): touchable {iv['touch']} frames, "
                  f"Boss_flag {iv['boss_flag']} frames, counts {cs[:6]}{'...' if len(cs) > 6 else ''}")
