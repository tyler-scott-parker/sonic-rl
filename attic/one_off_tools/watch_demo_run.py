"""Overnight watch numbers for a v4/v5 demo run. Usage: watch_d5.py [run] [baseline]
(defaults d5, d3). Frames as the context doc counts them: per episode steps - demo_frac*demo_len
(fresh: steps). Multi-stage runs: one pointer per demo, frames per stage; the baseline comparison is
stage 1's pointer at equal STAGE-1 frames (d4/d6 interference read)."""
import json, os, sys, time
REPO = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, REPO)
import demos                                     # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'd5'
BASE = sys.argv[2] if len(sys.argv) > 2 else 'd3'
_len = {}


def demo_len(did):
    if did not in _len:
        _len[did] = demos.load_demo(os.path.join(demos.DEMO_DIR, f'{did}.npz'))[1]['frames']
    return _len[did]


def walk(run, stop_at_stage1=None):
    """-> totals, per-stage frames, per-demo pointer state, fresh episodes, first fresh clear."""
    total, per_stage, ptr, fresh, first_clear = 0.0, {}, {}, [], None
    for line in open(os.path.join(os.environ.get('WATCH_LOG_DIR', REPO), f'ss_log_{run}.jsonl')):
        try:
            r = json.loads(line)
        except ValueError:
            continue                             # a line mid-write
        did = r.get('demo')
        f = r['steps'] - r['demo_frac'] * demo_len(did) if did else r['steps']
        if stop_at_stage1 is not None and r['stage'] == 1 and per_stage.get(1, 0) + f > stop_at_stage1:
            break
        total += f
        per_stage[r['stage']] = per_stage.get(r['stage'], 0) + f
        if did:
            # episodes start in [pointer, pointer + DEMO_BAND] and the pointer only moves
            # back, so the running min of demo_start is the pointer
            p = ptr.get(did)
            if p is None or r['demo_start'] < p['ckpt']:
                ptr[did] = dict(ckpt=r['demo_start'], frac=r['demo_frac'], since=total,
                                since_own=per_stage[r['stage']], stage=r['stage'],
                                hist=(p['hist'] if p else []) + [(round(total / 1e6, 2), round(100 * r['demo_frac']))])
        else:
            fresh.append(r)
            if r['success'] and first_clear is None:
                first_clear = (r['stage'], round(total / 1e6, 2), time.strftime('%m-%d %H:%M', time.localtime(r['t'])))
    return total, per_stage, ptr, fresh, first_clear


total, per_stage, ptr, fresh, first_clear = walk(RUN)
print(f'{RUN}: {total / 1e6:.2f}M frames' + ('' if len(per_stage) == 1 else
      ' (' + ', '.join(f's{s} {v / 1e6:.2f}M' for s, v in sorted(per_stage.items())) + ')'))
for did, p in sorted(ptr.items(), key=lambda kv: kv[1]['stage']):
    own = per_stage[p['stage']] - p['since_own']
    print(f'  stage {p["stage"]} start point {round(100 * p["frac"])}% (ckpt {p["ckpt"]}) since {p["since"] / 1e6:.2f}M '
          f'({(total - p["since"]) / 1e6:.2f}M frames unmoved, {own / 1e6:.2f}M of its own)')
    print(f'    moves (M frames, %): {p["hist"][-8:]}')
for s in sorted({r['stage'] for r in fresh}):
    last = [r for r in fresh if r['stage'] == s][-300:]
    allf = [r for r in fresh if r['stage'] == s]
    print(f'  fresh stage {s}, last {len(last)}: mean {sum(r["spheres"] for r in last) / len(last):.1f}/'
          f'{last[-1]["stage_total"]}, best {max(r["spheres"] for r in last)}, clears {sum(r["success"] for r in last)}'
          f' | all-time best {max(r["spheres"] for r in allf)}')
print(f'  first fresh clear (stage, M frames, time): {first_clear}')
if BASE != RUN and 1 in per_stage:
    _, bps, bptr, _, _ = walk(BASE, stop_at_stage1=per_stage[1])
    b = [p for p in bptr.values() if p['stage'] == 1][0]
    mine = [p for p in ptr.values() if p['stage'] == 1]
    gap = f', {RUN} {mine[0]["ckpt"] - b["ckpt"]:+d} ckpts vs it' if mine else ''
    print(f'  {BASE} at the same {per_stage[1] / 1e6:.2f}M stage-1 frames: start point {round(100 * b["frac"])}% '
          f'(ckpt {b["ckpt"]}){gap}')
