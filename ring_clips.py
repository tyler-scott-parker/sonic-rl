#!/usr/bin/env python3
"""
ring_clips.py — record short detour clips: from a spot on your own route to a giant ring the agent
never finds, so it can learn the way in (Tyler 2026-10-15: "Detour clips sounds excellent, as long as I
can resume near those areas and don't have to replay the whole level").

  python3 ring_clips.py                      # list the rings: where your run passes, the agent's record
  python3 ring_clips.py 4                    # record ring 4 from the point on your route nearest it
  python3 ring_clips.py 4 --back 20          # ... from 20 s earlier on your route (the way in may branch off before)
  python3 ring_clips.py 4 --speed 100        # anything else goes to record_play.py (speed, --no-sound)
  python3 ring_clips.py --export             # (once) build the start states from your recordings

Why: the ring share starts the agent where your route passes nearest each ring, but 12 of the 14 rings in
mg2's acts sit 200-360 px off your route, under a floor or behind a wall and off screen, and in 43 of 44
probe tries the agent never got closer than its first frame (attic/one_off_tools/ring_reach_probe.py).

Recording is record_play.py itself (same keys, rewind, speed, sound), started from the saved state of
your own run at that spot. Go into the giant ring, then press Enter to save and quit: the special stage
doesn't need playing. If getting back to your usual route from the ring isn't obvious, you can play on
(lose the stage quickly), get back onto your route, and press Enter then. The clip lands in
demos/clips/ with the ring it went into (checked by replaying it).

The start states (--export: every 5-15 s up to a minute back, for each ring) are written to
states/ring_clips/, from the start points the training env builds from your recordings, so they are
exactly frames of your run. CPU only.
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import time
import zlib

HOME = os.path.dirname(os.path.abspath(__file__))
RUN = 'mg2'
OUT_DIR = os.path.join('states', 'ring_clips')
INDEX = os.path.join(OUT_DIR, 'index.json')
CLIP_DIR = os.path.join('demos', 'clips')
BACKS = (0, 5, 10, 15, 20, 30, 45, 60)     # seconds earlier on his route
TAKES = 0.2                                # the agent takes a ring this often from its start: no clip needed
ENTERED_PX = 96                            # Sonic this close to a ring on the frame before a stage: went in


def fmt(frames):
    s = frames // 60
    return f'{s // 60}:{s % 60:02d}'


def ring_record(log_path):
    """{(zone, act, x, y): (tries, took)} from the run's ring-share episodes."""
    out = {}
    try:
        for line in open(log_path):
            if '"ring_share"' not in line:
                continue
            r = json.loads(line)
            k = tuple(r['ring_share'])
            n, t = out.get(k, (0, 0))
            out[k] = (n + 1, t + (list(k) in r.get('giant_rings', [])))
    except OSError:
        pass
    return out


def export():
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(RUN)
    os.environ.update(MG_LOG=os.devnull, CUDA_VISIBLE_DEVICES='')
    import numpy as np
    import coordinator as C
    env = C.SonicCoordinator(demo_prob=0.0)
    env.demo_info()
    rec = ring_record(os.path.join(HOME, f'mg_log_{RUN}.jsonl'))
    os.makedirs(os.path.join(HOME, OUT_DIR), exist_ok=True)
    rings = []
    for n, (seg, k, ring) in enumerate(env.ring_starts(), 1):
        z, a, rx, ry = ring
        r = seg['route']
        m = (r['zone'] == z) & (r['act'] == a)
        d = np.hypot(r['x'][m] - rx, r['y'][m] - ry)
        i = int(d.argmin())
        ts = [c['t'] for c in seg['ckpts']]
        starts = {}
        for back in BACKS:
            kk = next(j for j, t in enumerate(ts) if t >= ts[k] - 60 * back)
            name = f"ring{n:02d}_{seg['name']}_{rx}_{ry}_back{back:02d}.state.gz"
            with gzip.open(os.path.join(HOME, OUT_DIR, name), 'wb', compresslevel=6) as f:
                f.write(zlib.decompress(seg['ckpts'][kk]['state']))
            j = int(np.searchsorted(r['t'], ts[kk]))
            starts[str(back)] = dict(file=os.path.join(OUT_DIR, name), start_point=kk, t=int(ts[kk]),
                                     x=int(r['x'][min(j, len(r['x']) - 1)]), y=int(r['y'][min(j, len(r['y']) - 1)]))
        tries, took = rec.get(tuple(ring), (0, 0))
        rings.append(dict(n=n, act=seg['name'], act_key=seg['key'], ring=list(ring), demo=seg['demo'],
                          his_nearest=dict(x=int(r['x'][m][i]), y=int(r['y'][m][i]), t=int(r['t'][m][i]),
                                           px=round(float(d[i]))),
                          ring_start_point=k, tries=tries, took=took, starts=starts))
        print(f"  ring {n:2d} {seg['name']} ({rx},{ry}): {len(starts)} starts", flush=True)
    with open(os.path.join(HOME, INDEX), 'w') as f:
        json.dump(dict(run=RUN, made=time.strftime('%Y-%m-%d %H:%M:%S'), rings=rings), f, indent=1)
    print(f'wrote {len(rings)} rings to {INDEX}')


def load_index():
    try:
        return json.load(open(os.path.join(HOME, INDEX)))['rings']
    except OSError:
        sys.exit(f'no {INDEX} yet: run  python3 ring_clips.py --export  once (CPU, a few minutes)')


def clips_of(ring):
    return sorted(glob.glob(os.path.join(HOME, CLIP_DIR, f"ring{ring['n']:02d}_*.npz")))


def show(rings):
    print(f'Giant rings in {RUN}\'s acts. "agent" = its ring-share tries / times it went in.\n')
    for g in rings:
        z, a, rx, ry = g['ring']
        h = g['his_nearest']
        need = g['tries'] and g['took'] / g['tries'] >= TAKES
        print(f"  {g['n']:2d}  {g['act']}  ring at ({rx},{ry})  your run passes {h['px']} px away at "
              f"{fmt(h['t'])} into the act   agent {g['took']}/{g['tries']}"
              + ('  (finds it: no clip needed)' if need else '')
              + (f"   clips: {len(clips_of(g))}" if clips_of(g) else ''))
    print(f"\nRecord:  python3 ring_clips.py N [--back S]   S = {', '.join(map(str, BACKS))} s earlier on your route")


def stage_entry(path):
    """(frame, (x, y) just before, the giant ring nearest there, its distance) for each special
    stage a clip went into, by replaying it."""
    sys.path.insert(0, HOME)
    import coordinator as C
    import demos
    rings = C.GIANT_RINGS or C.giant_rings()
    actions, meta = demos.load_play(path)
    out, prev = [], {'info': None, 'stage': False}

    def on_frame(i, info):
        stage = C.frame_flags(info)[1]
        p = prev['info']
        if stage and not prev['stage'] and p is not None:
            x, y = int(p['x']), int(p['y'])
            cand = rings.get((int(p['zone']), int(p['act'])), ())
            best = min(cand, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2) if cand else None
            out.append((i, (x, y), best, math.hypot(best[0] - x, best[1] - y) if best else None))
        prev['info'], prev['stage'] = info, stage
    demos.replay_play(actions, meta, on_frame=on_frame)
    return out


def file_clip(path, g, back):
    """Check which ring a recording from ring g's start went into, label it, move it to demos/clips/."""
    sys.path.insert(0, HOME)
    import demos
    actions, meta = demos.load_play(path)
    ent = stage_entry(path)
    z, a, rx, ry = g['ring']
    hit = [e for e in ent if e[2] and tuple(e[2]) == (rx, ry) and e[3] <= ENTERED_PX]
    meta['clip'] = dict(ring=g['ring'], ring_n=g['n'], act=g['act'], act_key=g['act_key'], demo=g['demo'],
                        back_s=back, start_point=g['starts'][str(back)]['start_point'],
                        start_t=g['starts'][str(back)]['t'],
                        entered_at=hit[0][0] if hit else None,
                        stages=[dict(frame=e[0], at=list(e[1]), ring=list(e[2]) if e[2] else None,
                                     px=round(e[3]) if e[3] is not None else None) for e in ent])
    os.makedirs(os.path.join(HOME, CLIP_DIR), exist_ok=True)
    dest = os.path.join(HOME, CLIP_DIR, f"ring{g['n']:02d}_{g['act']}_{rx}_{ry}_{demos.demo_id(path)}.npz")
    demos.save_play(dest, actions, meta)
    os.remove(path)
    if hit:
        print(f"clip saved: {os.path.relpath(dest, HOME)}: went into ring {g['n']} at frame {hit[0][0]} "
              f"({hit[0][0] / 60:.1f} s), {len(actions)} frames in all")
    else:
        other = ', '.join(f"ring at {e[2]} ({e[3]:.0f} px)" for e in ent if e[2]) or 'none'
        print(f"clip saved: {os.path.relpath(dest, HOME)}, but it did NOT go into ring {g['n']} "
              f"(special stages entered: {other}). Keep it or delete it; training will only use clips that did.")
    return dest, bool(hit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ring', nargs='?', type=int, help='ring number from the list')
    ap.add_argument('--back', type=int, default=0, choices=BACKS, help='start this many seconds earlier on your route')
    ap.add_argument('--export', action='store_true', help='build the start states (once)')
    args, extra = ap.parse_known_args()
    os.chdir(HOME)
    if args.export:
        return export()
    rings = load_index()
    if args.ring is None:
        return show(rings)
    g = next((x for x in rings if x['n'] == args.ring), None)
    if g is None:
        sys.exit(f'no ring {args.ring}: 1..{len(rings)}')
    st = g['starts'][str(args.back)]
    z, a, rx, ry = g['ring']
    print(f"ring {g['n']}: {g['act']}, giant ring at ({rx},{ry}). You start {fmt(st['t'])} into the act at "
          f"({st['x']},{st['y']}), {args.back} s before your run passes nearest it.\n"
          f"Go into the ring, then Enter saves and quits (no need to play the stage). Esc twice quits without saving.",
          flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, os.path.join(HOME, 'record_play.py'), '--state', st['file']] + extra)
    sys.path.insert(0, HOME)
    import demos
    new = []
    for p in glob.glob(os.path.join(HOME, demos.DEMO_DIR, 'play_*.npz')):
        if os.path.getmtime(p) >= t0:
            try:
                if demos.load_play(p)[1].get('start_state') == st['file']:
                    new.append(p)
            except ValueError:
                pass
    if not new:
        print('nothing recorded')
        return
    if rc != 0:
        print(f'note: record_play.py exited with {rc} (its replay check); filing the clip anyway')
    file_clip(max(new, key=os.path.getmtime), g, args.back)


if __name__ == '__main__':
    main()
