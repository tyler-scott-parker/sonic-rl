"""
demos.py — the recorded-demo format, shared by record_demo.py (writes demos),
special_stage_env.py (starts episodes from them), train_ss_multistage.py
(checks them) and demo_progress.py (reports on them).

A demo is ONLY YOUR INPUTS: the stage, one agent action (0-6, the env's
ACTION_BUTTONS) per frame from that stage's start state, and some metadata.
The game is deterministic, so replaying the actions from the same start state
rebuilds the whole run, in whatever observation format the agent uses. No
emulator memory is stored, so unlike states/ a demo holds no ROM data.

The start state is pinned by the SHA1 of its file: re-capturing a stage (or
changing the action table) makes old demos refuse to load instead of silently
replaying into a different game.
"""
import os
import json
import hashlib

import numpy as np

DEMO_FORMAT = 1
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, 'demos')


def file_sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def actions_sha1(action_buttons):
    return hashlib.sha1(json.dumps(action_buttons).encode()).hexdigest()


def demo_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def save_demo(path, actions, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(meta, format=DEMO_FORMAT)
    np.savez(path, actions=np.asarray(actions, dtype=np.uint8),
             meta=np.array(json.dumps(meta)))


def load_demo(path):
    with np.load(path) as d:
        actions = d['actions'].astype(np.int64)
        meta = json.loads(str(d['meta']))
    if meta.get('format') != DEMO_FORMAT:
        raise ValueError(f'{path}: demo format {meta.get("format")}, '
                         f'this code reads {DEMO_FORMAT}')
    return actions, meta


def check_compatible(path, meta, sse):
    """Raise unless the demo still replays into the same game: same stage
    start state, same action table, env v2."""
    sf = sse.stage_file(meta['stage'])
    if file_sha1(sf) != meta['state_sha1']:
        raise ValueError(f'{path}: {sf} has changed since this demo was recorded')
    if actions_sha1(sse.ACTION_BUTTONS) != meta['actions_sha1']:
        raise ValueError(f'{path}: the action table has changed since recording')
    if not sse.V2:
        raise ValueError('demos need SS_ENV_VERSION >= 2')


def replay_check(path):
    """Replay a demo through the training env and confirm it reproduces the
    recorded run. Returns (ok, message). Uses one emulator; call it only in a
    process that has none open. Needs special_stage_env importable as env v2."""
    import special_stage_env as sse
    actions, meta = load_demo(path)
    try:
        check_compatible(path, meta, sse)
    except ValueError as e:
        return False, str(e)
    sf = sse.stage_file(meta['stage'])
    env = sse.SpecialStageEnv(state_path=sf)
    env._demos = []                                   # don't build demos in here
    env._log_episode = lambda *a, **k: None
    env._replaying = True                             # v6+: a recording skips the stall rule,
                                                      # as in the env's own demo replays (stage 6
                                                      # waits > 600 frames for its last sphere)
    try:
        env.reset(options={'stage_file': sf, 'replay': True})
        ended_at = None
        for i, a in enumerate(actions):
            _, _, done, trunc, _ = env._step_frames(int(a), 1)   # frame by frame
            if done or trunc:
                ended_at = i
                break
        spheres = env._spheres_collected
    finally:
        env.close()
    last = len(actions) - 1
    if meta['outcome'] in ('clear', 'death', 'timeout'):
        ok = ended_at == last and spheres == meta['spheres']
    else:                                             # quit: stopped mid-run
        ok = ended_at is None and spheres == meta['spheres']
    msg = (f'replay reproduces the run: {spheres} spheres, {meta["outcome"]} '
           f'after {len(actions)} frames' if ok else
           f'REPLAY MISMATCH: recorded {meta["spheres"]} spheres / '
           f'{meta["outcome"]} at frame {last}, replay got {spheres} spheres, '
           f'ended at {ended_at}')
    return ok, msg


# ── Main-game playthroughs (record_play.py -> coordinator.py) ────────────────
# Same idea — only the inputs — plus what the special-stage format lacks:
#   - the action TABLE itself (the 12-button mask per index), not just its
#     hash, so a later change to the agent's actions can't reinterpret an old
#     recording; the replay always uses the table it was recorded with
#   - the start state by its integration name + SHA1 of its bytes
#   - the acts the run passed through (coordinator.ActSplitter), so a replay
#     can prove it reproduces them frame for frame
PLAY_FORMAT = 1


def save_play(path, actions, meta):
    """Atomic: a crash mid-save leaves the previous file intact."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = dict(meta, kind='main_game', play_format=PLAY_FORMAT)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, actions=np.asarray(actions, dtype=np.uint8),
                 meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_play(path):
    with np.load(path) as d:
        actions = d['actions'].astype(np.int64)
        meta = json.loads(str(d['meta']))
    if meta.get('kind') != 'main_game' or meta.get('play_format') != PLAY_FORMAT:
        raise ValueError(f'{path}: not a main-game playthrough of format {PLAY_FORMAT} '
                         f'(kind {meta.get("kind")}, format {meta.get("play_format")})')
    return actions, meta


def play_start_state(meta):
    """The bytes the recording started from; raises if they have changed."""
    import coordinator
    data = coordinator.integration_state(meta['start_state'])
    if coordinator.sha1(data) != meta['start_sha1']:
        raise ValueError(f"start state {meta['start_state']} has changed since "
                         f"this playthrough was recorded")
    return data


def replay_play(actions, meta, env=None, on_frame=None):
    """Replay a playthrough exactly as record_play.py played it. Returns the
    ActSplitter. Uses one emulator — pass the process's own if it has one."""
    import coordinator
    own = env is None
    env = env or coordinator.make_retro()
    try:
        env.reset()
        env.em.set_state(play_start_state(meta))
        split = coordinator.ActSplitter()
        table = meta['action_table']
        for i, a in enumerate(actions):
            _, info = coordinator.emu_frame(env, table[a])
            split.feed(i, info)
            if on_frame:
                on_frame(i, info)
    finally:
        if own:
            env.close()
    return split


def replay_play_check(path, env=None):
    """Replay a playthrough and confirm it reproduces the recorded acts —
    every act's name, first frame and clear frame. Returns (ok, message)."""
    actions, meta = load_play(path)
    try:
        split = replay_play(actions, meta, env=env)
    except ValueError as e:
        return False, str(e)
    got = [(a['name'], a['start'], a['clear']) for a in split.summary()]
    want = [(a['name'], a['start'], a['clear']) for a in meta['acts']]
    if got != want:
        return False, f'REPLAY MISMATCH: recorded {want}, replay {got}'
    done = [f"{a['name']} {a['time']}" for a in meta['acts'] if a['clear'] is not None]
    return True, (f'replay reproduces the run: {len(actions)} frames; cleared '
                  f"{', '.join(done) if done else 'no act yet'}")
