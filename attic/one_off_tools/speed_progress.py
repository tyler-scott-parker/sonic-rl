"""Clear-time progress for a speed run (d8). Usage: speed_progress.py [run] [window]
Frames as the context doc counts them (per episode: steps - demo_frac x demo
length; fresh: steps). Fresh starts = episodes without a demo start. Prints
the last `window` (default 300) fresh starts — clear rate, median / p25 / best
clear frames — and the same read per 2M-frame bucket, so the trend shows."""
import json, os, statistics as st, sys
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO)
import demos                                     # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'd8'
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BUCKET = 2e6
_len = {}


def demo_len(did):
    if did not in _len:
        _len[did] = demos.load_demo(os.path.join(demos.DEMO_DIR, f'{did}.npz'))[1]['frames']
    return _len[did]


def read(rows):
    clears = [r['steps'] for r in rows if r['success']]
    if not rows:
        return 'no fresh starts yet'
    s = f'{len(rows)} fresh: clears {len(clears)} ({len(clears) / len(rows):.1%})'
    if clears:
        q = st.quantiles(clears, n=4) if len(clears) > 1 else [clears[0]] * 3
        s += f', median {st.median(clears):.0f} frames (p25 {q[0]:.0f}), best {min(clears)}'
    return s


total, fresh, buckets = 0.0, [], {}
for line in open(os.path.join(REPO, f'ss_log_{RUN}.jsonl')):
    try:
        r = json.loads(line)
    except ValueError:
        continue                                 # a line mid-write
    did = r.get('demo')
    total += r['steps'] - r['demo_frac'] * demo_len(did) if did else r['steps']
    if not did:
        fresh.append(r)
        buckets.setdefault(int(total // BUCKET), []).append(r)
print(f'{RUN}: {total / 1e6:.2f}M frames; last {WIN} {read(fresh[-WIN:])}')
for b, rows in sorted(buckets.items()):
    print(f'  {b * BUCKET / 1e6:4.0f}-{(b + 1) * BUCKET / 1e6:.0f}M: {read(rows)}')
