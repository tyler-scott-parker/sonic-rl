"""Small multiples for a multi-board demo run: per board, the curriculum start point
(running min of demo_start, as watch_demo_run.py reads it) against the run's total
frames, with fresh-start clears marked (x = when, label = the fastest's frames).
CPU, reads ss_log_<run>.jsonl only. Usage:
    plot_demo_run.py [run] [out.png]          # defaults d9, logs/<run>_progress.png"""
import json, os, sys
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402
import demos                                     # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'd9'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, 'logs', f'{RUN}_progress.png')
meta = {}


def demo_meta(did):
    if did not in meta:
        meta[did] = demos.load_demo(os.path.join(demos.DEMO_DIR, f'{did}.npz'))[1]
    return meta[did]


total, ptr, his, clears = 0.0, {}, {}, {}
for line in open(os.path.join(REPO, f'ss_log_{RUN}.jsonl')):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    s, did = int(r['stage']), r.get('demo')
    total += r['steps'] - r['demo_frac'] * demo_meta(did)['frames'] if did else r['steps']
    if did:
        his[s] = demo_meta(did)['frames']
        p = ptr.setdefault(s, [(0.0, 100.0)])
        if 100 * r['demo_frac'] < p[-1][1]:
            p.append((total / 1e6, 100 * r['demo_frac']))
    elif r['success']:
        clears.setdefault(s, []).append((total / 1e6, r['steps']))

boards = sorted(set(ptr) | set(clears))
cols = 7
rows = (len(boards) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4), sharex=True, sharey=True, squeeze=False)
for ax, s in zip(axes.flat, boards):
    p = ptr.get(s, [(0.0, 100.0)]) + [(total / 1e6, ptr.get(s, [(0, 100.0)])[-1][1])]
    ax.step([x for x, _ in p], [y for _, y in p], where='post', color='tab:blue', lw=1.5)
    c = clears.get(s, [])
    if c:
        ax.plot([x for x, _ in c], [3] * len(c), '|', color='tab:green', ms=8, mew=1.2)
    best = min((f for _, f in c), default=None)
    note = f'fresh clears {len(c)}' + (f', best {best}' if best else '')
    ax.set_title(f'board {s} (his {his.get(s, 0)})', fontsize=9)
    ax.text(0.03, 0.12, note, transform=ax.transAxes, fontsize=7,
            color='tab:green' if best and best < his.get(s, 0) else 'black')
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
for ax in axes.flat[len(boards):]:
    ax.axis('off')
for ax in axes[-1]:
    ax.set_xlabel('run frames (M)', fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel('start point (%)', fontsize=8)
fig.suptitle(f'{RUN}: curriculum start point per board (100% = recording end, 0% = fresh start); '
             f'green ticks = fresh clears; {total / 1e6:.1f}M frames', fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f'wrote {OUT} ({total / 1e6:.2f}M frames, {sum(len(v) for v in clears.values())} fresh clears)')
