"""The v7 prototype must leave env v4-v6 untouched: the same seeded corner
trajectories under git HEAD's special_stage_env.py and under the working
copy give identical observations and rewards.

  python3 attic/one_off_tools/check_v7_isolation.py 4 5 6     # ALL CHECKS PASS

Per version: d4's preset (v4 also with d7's compass), stages 1/5/10 from a
fresh start, seeded random corner choices for 400 steps each.
"""
import os, subprocess, sys, tempfile
import numpy as np

HOME = '/home/tyler/Projects/sonic-rl'
STAGES, STEPS = (1, 5, 10), 400


def record(out, version, pointer):
    os.environ.update(SS_ENV_VERSION=version, SS_USE_PIXELS='0', SS_PATCH_R='7', SS_DROP_POSITION='1',
                      SS_POINTER=pointer, SS_STAGES='1,5,10', SS_RESTART_PROB='0', SS_DEMOS='',
                      SS_DEMO_PROB='0', CUDA_VISIBLE_DEVICES='',
                      SS_LOG=os.path.join(tempfile.gettempdir(), f'check_iso_{os.getpid()}.jsonl'))
    import special_stage_env as sse
    env = sse.SpecialStageEnv()
    rng = np.random.default_rng(99)
    rams, grids, rews = [], [], []
    for st in STAGES:
        obs, _ = env.reset(options={'stage_file': sse.stage_file(st)})
        for _ in range(STEPS):
            rams.append(obs['ram']); grids.append(obs['grid'])
            obs, r, done, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            rews.append(r)
            if done or trunc:
                obs, _ = env.reset(options={'stage_file': sse.stage_file(st)})
    np.savez(out, ram=np.array(rams), grid=np.array(grids), rew=np.array(rews),
             n=env.action_space.n, envfile=sse.__file__)


def main():
    versions = sys.argv[1:] or ['4', '5', '6']
    tmp = tempfile.mkdtemp(prefix='check_iso_')
    old = os.path.join(tmp, 'old_env')
    os.mkdir(old)
    for name in os.listdir(HOME):
        if name != 'special_stage_env.py':
            os.symlink(os.path.join(HOME, name), os.path.join(old, name))
    with open(os.path.join(old, 'special_stage_env.py'), 'wb') as f:
        f.write(subprocess.run(['git', 'show', 'HEAD:special_stage_env.py'], cwd=HOME,
                               check=True, capture_output=True).stdout)
    ok = True
    for v in versions:
        for ptr in (('0', '1') if v == '4' else ('0',)):
            got = {}
            for tag, where in (('head', old), ('tree', HOME)):
                out = os.path.join(tmp, f'{tag}_{v}_{ptr}.npz')
                subprocess.run([sys.executable, '-c',
                                f'import sys; sys.path.insert(0, {where!r}); sys.argv=["x"]; '
                                f'import importlib.util as u; s=u.spec_from_file_location("ci", {__file__!r}); '
                                f'm=u.module_from_spec(s); s.loader.exec_module(m); m.record({out!r}, {v!r}, {ptr!r})'],
                               cwd=where, check=True)
                got[tag] = np.load(out)
            h, t = got['head'], got['tree']
            same = (h['rew'].shape == t['rew'].shape and np.array_equal(h['rew'], t['rew'])
                    and np.array_equal(h['ram'], t['ram']) and np.array_equal(h['grid'], t['grid'])
                    and int(h['n']) == int(t['n']))
            ok &= same
            print(f'  {"ok  " if same else "FAIL"} v{v} pointer={ptr}: {len(t["rew"])} steps, {int(t["n"])} actions, '
                  f'obs + rewards identical to git HEAD ({str(h["envfile"]).split("/")[-2]} vs {str(t["envfile"]).split("/")[-2]})',
                  flush=True)
    print('\nALL CHECKS PASS' if ok else '\nCHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
