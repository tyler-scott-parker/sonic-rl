import os, sys, json
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO); os.chdir(REPO)
DID = 'board14_2921_96f1dc25'
_src = open('train_ss_multistage.py').read(); _ns = {}
exec(_src[_src.index('ALL_14 ='):_src.index('# ── Shared config')], _ns)
os.environ.update(_ns['RUNS']['d10']['obs'], SS_STAGES='14', SS_RESTART_PROB='0', SS_DEMO_MIX='0',
                  SS_LOG=os.devnull, SS_DEMOS=f'routes/curriculum/{DID}.npz', SS_DEMO_PROB='1.0', CUDA_VISIBLE_DEVICES='')
import special_stage_env as sse, demos as demo_io
env = sse.SpecialStageEnv(); env._log_episode = lambda *a, **k: None
actions, meta = demo_io.load_demo(f'routes/curriculum/{DID}.npz')
env.reset(options={'stage_file': sse.stage_file(meta['stage']), 'replay': True})
NAMES = {0: '-', 1: 'B', 2: 'L', 3: 'R'}
prev = None
out = []
for f, a in enumerate(actions):
    _, _, done, trunc, info = env._step_frames(int(a), 1)
    ram = env._ram()
    cell = env._cell(int(info.get('ss_x', 0)), int(info.get('ss_y', 0)))
    jump = int(ram[sse.ADDR_JUMPING]) != 0
    got = env._spheres_collected
    if 1340 <= f <= 1560:
        key = (cell, jump, got, int(a))
        if key != prev:
            out.append(f"f{f} cell(r{cell[0]},c{cell[1]}) {'AIR' if jump else 'gnd'} btn {NAMES.get(int(a), a)} spheres {got}")
            prev = key
    if done or trunc:
        break
print('\n'.join(out))
