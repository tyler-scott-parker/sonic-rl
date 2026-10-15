#!/usr/bin/env python3
"""check_ring_clips.py — do a run's ring clips (MG_RING_CLIPS, RUNS 'ring_clips') start ring-share episodes
along Tyler's detours into the giant rings, honestly paid and out of the curriculum's way?

    python3 check_ring_clips.py [RUN]          # default mg2

1. the clips: RUNS 'ring_clips' resolves to the clips that went into their ring; each of an act in the
   run builds (its replay enters a stage on the recorded frame, else the build raises), with start points
   every CLIP_EVERY frames of its last CLIP_SPAN up to CLIP_BEFORE before he went in
2. every start point replays: his remaining clip frames from that point's saved state enter the stage on
   the recorded frame
3. route progress: each point's t never falls along the clip and starts at or after the clip's start; a
   reset at a point and one step of no input pays ~one step's frames, not the clip's ground
4. 60,000 drawn starts (episode draws): the ring share as set, every draw for a ring with a clip starts on
   one of its points (all points drawn, none for rings without one)
5. a clip reset runs and logs ring_share + ring_clip; its result says clip, so the curriculum's at-pointer
   rule skips it even when its 'start' equals the pointer (train_main.s ActCurriculumCallback itself)
6. MG_RING_CLIPS empty: no clip starts
CPU only; the episode log goes to a temp file. Ends with ALL CHECKS PASS or the failures.
"""
import collections
import json
import os
import random
import sys
import tempfile

HOME = os.path.dirname(os.path.abspath(__file__))
N = 60_000
bad = []


def check(ok, what):
    print(('  ok   ' if ok else '  FAIL ') + what)
    if not ok:
        bad.append(what)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'mg2'
    sys.path.insert(0, HOME)
    import train_main
    train_main.configure_env(run)
    log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    os.environ.update(MG_LOG=log, CUDA_VISIBLE_DEVICES='', MG_SS='truncate')
    import coordinator as C
    import demos
    C.MG_BALANCE_TIME = False                       # the episode draws; check_balance_time.py does the time
    cfg = train_main.RUNS[run]
    files = train_main.ring_clip_files(cfg.get('ring_clips'))

    print('1. the clips')
    check(len(files) > 0 and C.MG_RING_CLIPS == files, f"{len(files)} clips that went in, from {cfg.get('ring_clips')}")
    env = C.SonicCoordinator()
    env.demo_info()
    cur = json.load(open(os.path.join(HOME, f'main_{run}_curriculum.json')))
    ptrs = {k: v['ptr'] for k, v in cur.items()}
    env.set_demo_pointers(ptrs)
    in_run = {s['key'] for s in env._segs}
    want = {}
    for p in files:
        c = demos.load_play(p)[1]['clip']
        if c['act_key'] in in_run:
            want[demos.demo_id(p)] = c
    built = {cl['name']: (ring, cl) for ring, cls in env._clips.items() for cl in cls}
    check(set(built) == set(want), f'{len(built)} built, of the {len(want)} in this run\'s acts')
    for name, (ring, cl) in sorted(built.items()):
        c = want[name]
        frames = [p['frame'] for p in cl['pts']]
        ok = (tuple(c['ring']) == ring and all(f % C.CLIP_EVERY == 0 for f in frames)
              and frames[0] >= max(C.CLIP_EVERY, c['entered_at'] - C.CLIP_SPAN) and frames[-1] <= c['entered_at'] - C.CLIP_BEFORE)
        check(ok, f"ring {c['ring_n']} ({c['act']}): {len(frames)} points, frames {frames[0]}..{frames[-1]}, "
                  f"he went in at {c['entered_at']}")

    print('\n2. every start point replays into the ring')
    em = env.base_env.em
    for name, (ring, cl) in sorted(built.items()):
        actions, meta = demos.load_play(next(p for p in files if demos.demo_id(p) == name))
        table, end = meta['action_table'], cl['entered_at']
        fails = 0
        for pt in cl['pts']:
            env.base_env.reset()
            em.set_state(C.zlib.decompress(pt['state']))
            info = None
            for i in range(pt['frame'] + 1, end + 1):
                _, info = C.emu_frame(env.base_env, table[actions[i]])
                if C.frame_flags(info)[1]:
                    break
            fails += not (info is not None and C.frame_flags(info)[1] and i == end)
        check(fails == 0, f"{name}: {len(cl['pts'])} points, {fails} that don't enter at frame {end}")

    print('\n3. route progress')
    for name, (ring, cl) in sorted(built.items()):
        ts = [p['t'] for p in cl['pts']]
        c = want[name]
        check(all(b >= a for a, b in zip(ts, ts[1:])) and ts[0] >= c['start_t'],
              f"{name}: t {ts[0]}..{ts[-1]} along the clip (his route start {c['start_t']})")
    # ring-share resets until clip starts come up; one step of no input each
    step_r = []
    C.MG_BOSS_SHARE, C.MG_RING_SHARE, C.MG_FRONT_SHARE = 0.0, 1.0, 0.0
    seen = 0
    for _ in range(400):
        env.reset()
        if env._clip is None:
            continue
        _, r, term, trunc, _ = env.step(0)
        step_r.append(r)
        seen += 1
        if seen >= 40:
            break
    check(seen >= 20 and max(abs(r) for r in step_r) <= 4 * C.FRAME_SKIP + 1,
          f'{seen} clip resets: one step of no input pays {min(step_r):+.1f}..{max(step_r):+.1f} (one step = {C.FRAME_SKIP} frames)')

    print(f'\n4. {N:,} drawn starts')
    C.MG_BOSS_SHARE, C.MG_RING_SHARE, C.MG_FRONT_SHARE = (cfg.get('boss_share', 0), cfg.get('ring_share', 0),
                                                         cfg.get('front_share', 0))
    random.seed(13)
    ring_n, clip_n, by_pt, wrong = 0, 0, collections.Counter(), 0
    for _ in range(N):
        env._clip_pick = None
        seg, k, b, ring, front = env._pick({})
        if ring:
            ring_n += 1
        if env._clip_pick is not None:
            cl, j = env._clip_pick
            clip_n += 1
            by_pt[(cl['name'], j)] += 1
            wrong += not ring or tuple(ring) not in env._clips or cl not in env._clips[tuple(ring)] or seg is not cl['seg']
        elif ring and tuple(ring) in env._clips:
            wrong += 1
        env._clip_pick = None
    exp = C.MG_RING_SHARE
    check(abs(ring_n / N - exp) < 0.01, f'ring share {ring_n / N:.1%} of the draws (set {exp:.0%})')
    n_pts = sum(len(cl['pts']) for cls in env._clips.values() for cl in cls)
    check(wrong == 0 and clip_n > 0, f'{clip_n} clip starts, all on their ring\'s clip; ring draws for clip rings never on his route ({wrong} wrong)')
    check(len(by_pt) == n_pts, f'all {n_pts} points drawn ({len(by_pt)} seen)')

    print('\n5. a clip reset runs and is kept out of the curriculum')
    C.MG_BOSS_SHARE, C.MG_RING_SHARE, C.MG_FRONT_SHARE = 0.0, 1.0, 0.0
    for _ in range(400):
        env.reset()
        if env._clip is not None:
            break
    clip, k0, key0 = env._clip, env._k, env._seg['key']
    done, n = False, 0
    while not done and n < 4000:
        _, _, term, trunc, info = env.step(random.randrange(len(C.ACTION_BUTTONS)))
        done, n = term or trunc, n + 1
    with open(log) as f:
        r = json.loads(f.read().splitlines()[-1])
    res = info['mg_result']
    check(done and r.get('ring_clip') == list(clip) and r.get('ring_share') and res.get('clip') is True,
          f"{clip[0]} point {clip[1]}/{clip[2]}: {res['outcome']} after {res['frames']} frames, logged ring_clip {r.get('ring_clip')}")
    cb = train_main.ActCurriculumCallback.__new__(train_main.ActCurriculumCallback)
    cb.ptr, cb.recent = {key0: k0}, {key0: collections.deque(maxlen=train_main.CURR_WINDOW)}
    cb.at_pointer, cb.band, cb.min_n, cb.successes = True, 4, 32, 8
    cb.locals = {'infos': [{'mg_result': dict(res, start=k0)}]}
    cb._on_step()
    check(len(cb.recent[key0]) == 0, f'the curriculum skips it even with start == its pointer ({k0})')
    cb.locals = {'infos': [{'mg_result': dict(res, start=k0, clip=False)}]}
    cb._on_step()
    check(len(cb.recent[key0]) == 1, '... and counts the same result without the clip mark (the test can fail)')

    print('\n6. MG_RING_CLIPS empty')
    env._clips = {}
    C.MG_RING_SHARE = 1.0
    any_clip = False
    for _ in range(2000):
        env._clip_pick = None
        env._pick({})
        any_clip |= env._clip_pick is not None
    check(not any_clip, 'no clip starts')
    os.unlink(log)
    print('\nALL CHECKS PASS' if not bad else f'\n{len(bad)} FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
