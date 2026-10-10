"""Frames in Tyler's recordings where more than one blue sphere vanished at
once (the outline trick turns the enclosed blues into rings)."""
import os, sys, glob
HOME = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, HOME)
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read(); ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
os.environ.update(ns['RUNS']['d4']['obs'], SS_ENV_VERSION='6', SS_STAGES=','.join(map(str, range(1, 15))),
                  SS_RESTART_PROB='0', SS_DEMOS='', SS_DEMO_PROB='0', SS_LOG=os.environ['SP'] + '/loops.jsonl')
import special_stage_env as sse, demos as demo_io
env = sse.SpecialStageEnv()
tot_events = tot_conv = 0
for path in sorted(glob.glob(os.path.join(HOME, 'demos', 'stage*_*.npz')), key=lambda p: int(os.path.basename(p)[5:].split('_')[0])):
    actions, meta = demo_io.load_demo(path)
    env.reset(options={'stage_file': sse.stage_file(meta['stage']), 'replay': True})
    prev, drops = None, []
    for j, a in enumerate(actions):
        _, _, done, trunc, info = env._step_frames(int(a), 1)
        b = info.get('blue_spheres_remaining')
        if prev is not None and b is not None and prev - b > 1:
            drops.append((j, prev - b))
        prev = b
        if done or trunc:
            break
    conv = sum(d - 1 for _, d in drops)
    tot_events += len(drops); tot_conv += conv
    print(f"stage {meta['stage']:2d}: {meta['spheres']:3d} spheres, {len(actions):5d} frames; multi-sphere frames {len(drops)}"
          + (f" (sizes {[d for _, d in drops]}, {conv} spheres converted, not touched)" if drops else ''), flush=True)
print(f'ALL: {tot_events} multi-sphere frames, {tot_conv} spheres converted')
