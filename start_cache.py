"""
start_cache.py — start points built from recordings, kept on disk between trainer starts.

Tyler 2026-10-16 ("Go ahead and build the cache"): start-up took ~6 min for mg2 and ~4 for d10,
nearly all of it replaying his recordings: once to check them in the trainer, then again in each
of the 8 workers to build the start points. The replays are deterministic, so their result is
kept here and loaded instead.

A cache file is a pure function of its key: the sha1 of everything its build reads (the
recordings' bytes, the source of the modules that replay them, the process's MG_/SS_ settings,
the game integration incl. the ROM, and the emulator/numpy/cv2/Python versions). Any change gives
a new key, so a stale file is never read; unused files are pruned after PRUNE_DAYS. Only a
trainer's start-up writes (coordinator.prebuild / special_stage_env.prebuild_demos, after the
replay was verified); workers and tools read, and build as before when there's no file.
"""
import hashlib
import importlib.metadata
import os
import pickle
import sys
import time

HOME = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HOME, 'start_cache')
PRUNE_DAYS = 7
_FILE_SHA1 = {}                                    # path -> (size, mtime, sha1): one read per process
_COMMON = None


def file_sha1(path):
    st = os.stat(path)
    got = _FILE_SHA1.get(path)
    if got is None or got[:2] != (st.st_size, st.st_mtime_ns):
        with open(path, 'rb') as f:
            got = (st.st_size, st.st_mtime_ns, hashlib.sha1(f.read()).hexdigest())
        _FILE_SHA1[path] = got
    return got[2]


def _common():
    """What every build depends on: the game integration (ROM, data.json, ...) and the versions."""
    global _COMMON
    if _COMMON is None:
        import cv2
        import numpy as np
        import stable_retro
        integ = os.path.dirname(stable_retro.data.get_file_path('SonicAndKnuckles3-Genesis-v0', 'data.json'))
        _COMMON = [('integration', sorted((f, file_sha1(os.path.join(integ, f))) for f in os.listdir(integ))),
                   ('versions', importlib.metadata.version('stable-retro'), np.__version__, cv2.__version__,
                    sys.version_info[:3], pickle.HIGHEST_PROTOCOL)]
    return _COMMON


def key(kind, sources, files, env_prefix, exclude=(), extra=()):
    """The cache key: kind, the source files (repo-relative), the input files' bytes, every
    environment variable starting with env_prefix except those in exclude, and extra (any repr-able
    values, e.g. constructor arguments)."""
    h = hashlib.sha1()
    parts = [kind,
             [(s, file_sha1(os.path.join(HOME, s))) for s in sources],
             [(f, file_sha1(f)) for f in files],
             sorted((k, v) for k, v in os.environ.items() if k.startswith(env_prefix) and k not in exclude),
             list(extra), _common()]
    h.update(repr(parts).encode())
    return h.hexdigest()


def _path(kind, k):
    return os.path.join(DIR, f'{kind}_{k}.pkl')


def exists(kind, k):
    return os.path.exists(_path(kind, k))


def load(kind, k):
    """The object saved under this key, or None. A hit refreshes the file's time (pruning)."""
    p = _path(kind, k)
    try:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    try:
        os.utime(p)
    except OSError:
        pass
    return obj


def save(kind, k, obj):
    """Atomic (a crash leaves no half file); prunes files unused for PRUNE_DAYS."""
    os.makedirs(DIR, exist_ok=True)
    p = _path(kind, k)
    tmp = f'{p}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    cutoff = time.time() - PRUNE_DAYS * 86400
    for name in os.listdir(DIR):
        q = os.path.join(DIR, name)
        try:
            if os.path.getmtime(q) < cutoff:
                os.remove(q)
        except OSError:
            pass
    return p
