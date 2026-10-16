#!/usr/bin/env python3
"""check_stage_weights.py — does SS_STAGE_WEIGHTS draw boards in proportion to their weights, for fresh starts
and demo (route) starts, and leave the even draw alone when unset?

    python3 check_stage_weights.py [WEIGHTS] [N]      # default '14:6', 4000 resets

d10's own settings (env v7, the book's routes as demos, its demo_prob); weights relative to 1 for boards not
listed. Each kind of start: every board's share within 4 sigma of weight / total. CPU only.
"""
import collections, os, sys
HOME = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[1] if len(sys.argv) > 1 else '14:6'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read()
ns = {'__name__': 'x', '__file__': os.path.join(HOME, 'train_ss_multistage.py')}
exec(compile(src[:src.index('# ── Argument parsing MUST happen')], 'train_ss_multistage.py', 'exec'), ns)
cfg = ns['RUNS']['d10']
demos = ns['demo_paths'](cfg)
os.environ.update(cfg['obs'], SS_STAGES=','.join(map(str, cfg['stages'])), SS_RESTART_PROB='0',
                  SS_DEMOS=','.join(demos), SS_DEMO_PROB=str(cfg['demo_prob']), SS_LOG=os.devnull,
                  SS_STAGE_WEIGHTS=W, CUDA_VISIBLE_DEVICES='')
sys.path.insert(0, HOME)
import special_stage_env as sse
bad = []
w = {s: sse.STAGE_WEIGHTS.get(s, 1.0) for s in cfg['stages']}
print(f'weights {sse.STAGE_WEIGHTS} -> expected shares ' + ', '.join(f'{s}: {v / sum(w.values()):.1%}' for s, v in w.items() if v != 1.0)
      + f', others {1 / sum(w.values()):.1%} each')
env = sse.SpecialStageEnv()
env._log_episode = lambda *a, **k: None
cnt = {'demo': collections.Counter(), 'fresh': collections.Counter()}
for i in range(N):
    env.reset()
    kind = 'demo' if getattr(env, '_demo_episode', None) else 'fresh'
    cnt[kind][int(env._cur_stage)] += 1
for kind, c in cnt.items():
    n = sum(c.values())
    worst = 0.0
    for s in cfg['stages']:
        p = w[s] / sum(w.values())
        z = abs(c[s] / n - p) / (p * (1 - p) / n) ** 0.5
        worst = max(worst, z)
    ok = worst < 4
    if not ok:
        bad.append(kind)
    print(f"  {'ok  ' if ok else 'FAIL'} {kind} starts ({n}): board 14 {c[14] / n:.1%}, board 1 {c[1] / n:.1%}; worst {worst:.1f} sigma")
print('ALL CHECKS PASS' if not bad else f'FAILED: {bad}')
sys.exit(1 if bad else 0)
