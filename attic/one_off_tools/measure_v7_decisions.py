"""Env v7's two open questions, settled by measurement (rules agreed with Tyler
2026-10-10 before running: context doc, Pre-Committed Criteria, "Env v7 open
questions"). Run on the v7 prototype, d7's obs preset + v7, all 14 stages:

  nice -n 19 python3 attic/one_off_tools/measure_v7_decisions.py [out.json]

A. Up timing: at every back-restart decision (running backwards after a
   bounce) in his 14 recordings and in random play, Up-now vs Up-late, then
   each of straight / left / right / jump. Outcome = cell + heading at the
   decision after that, spheres, how the episode ended (if it did).
B1. Inferability: random-play decisions -> features the agent has (ram obs,
   its last action, the patch's centre cell) vs the "turn window still
   ahead" bit. MLP + lookup table, held-out episodes.
B2. Pointless decisions: at bumper-entry decisions and at decisions with a
   turn already queued, every action from the same snapshot, then straight
   at the next decision; does any action end somewhere other than straight?
"""
import os, sys, glob, json, copy, random, inspect, textwrap, collections, time
import numpy as np

HOME = '/home/tyler/Projects/sonic-rl'
sys.path.insert(0, HOME)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, 'attic', 'one_off_tools', 'measure_v7_decisions_out.json')
src = open(os.path.join(HOME, 'train_ss_multistage.py')).read(); ns = {}
exec(src[src.index('ALL_14 ='):src.index('# ── Shared config')], ns)
os.environ.update(ns['RUNS']['d7']['obs'], SS_ENV_VERSION='7', SS_STAGES=','.join(map(str, range(1, 15))),
                  SS_RESTART_PROB='0', SS_DEMOS='', SS_DEMO_PROB='0', CUDA_VISIBLE_DEVICES='',
                  SS_LOG=f'/tmp/claude-1000/measure_v7_{os.getpid()}.jsonl')
import special_stage_env as sse     # noqa: E402
import demos as demo_io             # noqa: E402
assert sse.V7, 'needs the v7 prototype in the working tree'

# ── instrumentation (copies of the env's methods with markers; logic unchanged)
sse.UP_NOW = [False]
s = inspect.getsource(sse.SpecialStageEnv._step_corner)
o = "late = presses[0] in ('late-jump', 'up')"
assert s.count(o) == 1
s = s.replace(o, o + " and not (presses[0] == 'up' and UP_NOW[0])")
exec(textwrap.dedent(s), vars(sse))
sse.SpecialStageEnv._step_corner = sse._step_corner
s = inspect.getsource(sse.SpecialStageEnv._at_decision)
for o, n in [("self._v7_pending = True\n            elif speed < 0", "self._v7_pending = True; self._sub = 'fwd-after-up'\n            elif speed < 0"),
             ("self._v7_pending = True\n            elif self._v7_air", "self._v7_pending = True; self._sub = 'back-restart'\n            elif self._v7_air"),
             ("self._v7_pending = True     # landed", "self._v7_pending = True; self._sub = 'landing'  # landed"),
             ("return True                     # Up needs", "self._kind = self._sub; return True  # Up needs"),
             ("return off < 0 if up else off >= v", "self._kind = 'normal'; return off < 0 if up else off >= v")]:
    assert s.count(o) == 1, o
    s = s.replace(o, n)
exec(textwrap.dedent(s), vars(sse))
_patched = sse._at_decision


def at_decision(self, info):
    prev = self._prev_xy
    r = _patched(self, info)
    if r:   # the "window still ahead" bit, along the actual motion (the v4 rule)
        x, y = int(info['ss_x']), int(info['ss_y'])
        mx = (x - prev[0] + 4096) % 8192 - 4096
        my = (y - prev[1] + 4096) % 8192 - 4096
        cell = self._cell(x, y)
        ox = (x - (cell[1] << 8) + 4096) % 8192 - 4096
        oy = (y - (cell[0] << 8) + 4096) % 8192 - 4096
        off, up, v = ((ox, mx > 0, abs(mx)) if abs(mx) >= abs(my) else (oy, my > 0, abs(my)))
        self._bit = int(off < 0 if up else off >= v)
    return r


sse.SpecialStageEnv._at_decision = at_decision
env = sse.SpecialStageEnv()
env._kind = env._sub = None
env._bit = 1
A_ = {n: i for i, n in enumerate(sse.CORNER_ACTIONS)}
UP, STRAIGHT = A_['up'], A_['straight']
SHARED = {'base_env', '_demos', 'action_space', 'observation_space', '_original_grids', '_actions'}


def snapshot():
    return (env.base_env.em.get_state(),
            {k: v if k in SHARED else copy.deepcopy(v) for k, v in env.__dict__.items()})


def restore(snap):
    env.base_env.em.set_state(snap[0])
    env.__dict__.update({k: v if k in SHARED else copy.deepcopy(v) for k, v in snap[1].items()})


def step(a):
    _, _, done, trunc, info = env.step(a)
    return done or trunc, info


def ending(done):
    if not done:
        return ''
    return 'clear' if env._board_cleared else 'stall' if env._stalled else 'red'


def where():
    return (env._decided, env._quadrant(int(env._ram()[sse.ADDR_HEADING])), env._spheres_collected)


# ── A. Up-now vs Up-late from one back-restart decision
A_rows, A_yard = [], []


def up_trial(tag):
    S, steps0 = snapshot(), env.steps
    res = {}
    was = env._replaying
    env._replaying = False
    for var in ('late', 'now'):
        restore(S); env._replaying = False
        sse.UP_NOW[0] = var == 'now'
        done, _ = step(UP)
        sse.UP_NOW[0] = False
        f1, kind1 = env.steps - steps0, env._kind
        if done:
            res[var] = dict(f_up=f1, outs={'*': [list(map(str, where())), ending(True), f1]})
            continue
        U = snapshot()
        outs = {}
        for a in ('straight', 'left', 'right', 'jump'):
            restore(U)
            done, _ = step(A_[a])
            outs[a] = [list(map(str, where())), ending(done), env.steps - steps0]
        res[var] = dict(f_up=f1, kind_after_up=kind1, outs=outs)
    if len(A_yard) < 20:      # snapshot yardstick: the same trial twice
        restore(S); env._replaying = False
        done, _ = step(UP)
        if not done:
            done, _ = step(STRAIGHT)
        A_yard.append([list(map(str, where())), ending(done), env.steps - steps0]
                      == res['late']['outs'].get('straight', res['late']['outs'].get('*')))
    restore(S); env._replaying = was
    A_rows.append(dict(src=tag, stage=env._cur_stage, **res))


# ── B2. every action at a bumper-entry / queued-turn decision
B2_rows = collections.defaultdict(list)


def all_actions(kind):
    S = snapshot()
    outs = {}
    for a, i in A_.items():
        restore(S)
        done, _ = step(i)
        if not done:
            done, _ = step(STRAIGHT)
        outs[a] = [list(map(str, where())), ending(done)]
    restore(S)
    B2_rows[kind].append(dict(stage=env._cur_stage, differ=sorted(a for a in outs if outs[a] != outs['straight'])))


t0 = time.time()
# his recordings: real bounces
for path in sorted(glob.glob(os.path.join(HOME, 'demos', 'stage*_*.npz'))):
    actions, meta = demo_io.load_demo(path)
    env.reset(options={'stage_file': sse.stage_file(meta['stage']), 'replay': True})
    env._replaying = True
    for a in actions:
        _, _, done, trunc, info = env._step_frames(int(a), 1)
        if done or trunc:
            break
        if env._at_decision(info) and env._kind == 'back-restart':
            up_trial(f"his s{meta['stage']}")
    env._replaying = False
print(f'A: {len(A_rows)} back-restart decisions in his recordings ({time.time() - t0:.0f}s)', flush=True)

# random play: B1 features, A bounces, B2 states
CAP_A, CAP_B2, EPS_B1 = (3, 3, 40) if os.environ.get('MEASURE_QUICK') else (300, 200, 700)
B1 = []                       # (episode, features..., bit, kind)
rng = random.Random(7)
ep = 0
while ep < 3000 and (ep < EPS_B1 or sum(r['src'] == 'random' for r in A_rows) < CAP_A
                      or min(len(B2_rows['bumper-entry']), len(B2_rows['queued'])) < CAP_B2):
    stage = ep % 14 + 1
    obs, _ = env.reset(options={'stage_file': sse.stage_file(stage)})
    env._kind, last = 'start', None
    while True:
        if ep < EPS_B1:
            g = obs['grid'][obs['grid'].shape[0] // 2, obs['grid'].shape[1] // 2]
            B1.append([ep] + list(map(float, obs['ram'])) + [last if last is not None else -1]
                      + list(map(float, g)) + [env._bit, env._kind])
        if env._kind == 'back-restart' and sum(r['src'] == 'random' for r in A_rows) < CAP_A:
            up_trial('random')
        on_bumper = env._decided is not None and int(env._original_grids[env._cur_stage][env._decided]) == 3
        if on_bumper and len(B2_rows['bumper-entry']) < CAP_B2:
            all_actions('bumper-entry')
        if int(env._ram()[0xE42A]) != 0 and len(B2_rows['queued']) < CAP_B2:
            all_actions('queued')
        last = rng.randrange(len(sse.CORNER_ACTIONS))
        obs, _, done, trunc, _ = env.step(last)
        if done or trunc:
            break
    ep += 1
    if ep % 100 == 0:
        print(f'  {ep} episodes, A {len(A_rows)}, B1 {len(B1)}, B2 bumper {len(B2_rows["bumper-entry"])} '
              f'queued {len(B2_rows["queued"])} ({time.time() - t0:.0f}s)', flush=True)

# ── analysis A
def outcome_set(r, var):
    return {(tuple(o[0][:2]), o[1]) for o in r[var]['outs'].values()}


lost, saved, rows_his = 0, [], [r for r in A_rows if r['src'] != 'random']
detail = []
for r in A_rows:
    L, N = outcome_set(r, 'late'), outcome_set(r, 'now')
    miss = L - N
    lost += bool(miss)
    for a in ('straight', 'left', 'right', 'jump'):
        ol, on = r['late']['outs'].get(a), r['now']['outs'].get(a)
        if ol and on and ol[0][:2] == on[0][:2] and ol[1] == on[1]:
            saved.append(ol[2] - on[2])
    detail.append(dict(src=r['src'], stage=r['stage'], lost=sorted(map(str, miss)),
                       f_up_late=r['late']['f_up'], f_up_now=r['now']['f_up']))
nA = len(A_rows)
print(f'\nA. Up timing: {nA} bounces ({len(rows_his)} his, {nA - len(rows_his)} random); snapshot yardstick '
      f'{sum(A_yard)}/{len(A_yard)} identical')
print(f'   Up-now loses an outcome Up-late reaches in {lost} ({100 * lost / max(1, nA):.1f}%)')
if saved:
    print(f'   frames saved by Up-now where both reach the same outcome: mean {np.mean(saved):.2f}, '
          f'median {np.median(saved):.0f}, min {min(saved)}, max {max(saved)} (n={len(saved)})')
for d in detail:
    if d['src'] != 'random':
        print(f'   {d["src"]}: up-step frames late {d["f_up_late"]} / now {d["f_up_now"]}; lost {d["lost"] or "nothing"}')

# ── analysis B1
import torch                       # noqa: E402
data = [r for r in B1]
eps = np.array([r[0] for r in data])
nram = len(obs['ram'])
X_ram = np.array([r[1:1 + nram] for r in data], dtype=np.float32)
last = np.array([r[1 + nram] for r in data], dtype=np.int64)
G = np.array([r[2 + nram:-2] for r in data], dtype=np.float32)
y = np.array([r[-2] for r in data], dtype=np.int64)
kinds = np.array([r[-1] for r in data])
onehot = np.eye(len(sse.CORNER_ACTIONS) + 1, dtype=np.float32)[last + 1]
X = np.concatenate([X_ram, onehot, G], axis=1)
test = np.isin(eps, np.unique(eps)[::5])          # every 5th episode held out
mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
Xn = (X - mu) / sd
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
                          torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
Xt, yt = torch.tensor(Xn[~test]), torch.tensor(y[~test])
for epoch in range(60):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        loss = torch.nn.functional.cross_entropy(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred = net(torch.tensor(Xn[test])).argmax(1).numpy()
acc_mlp = (pred == y[test]).mean()
# lookup table: (last action, heading quadrant, speed, jumping, centre cell)
def key(i):
    quad = int(round(np.arctan2(X_ram[i, 4], X_ram[i, 5]) / (np.pi / 2))) % 4   # sin, cos of heading
    return (int(last[i]), quad, round(float(X_ram[i, 1]), 2), tuple(G[i].round(2)))
table = collections.defaultdict(collections.Counter)
for i in np.flatnonzero(~test):
    table[key(i)][int(y[i])] += 1
glob_major = int(np.bincount(y[~test]).argmax())
pt = np.array([table[key(i)].most_common(1)[0][0] if table[key(i)] else glob_major for i in np.flatnonzero(test)])
acc_tab = (pt == y[test]).mean()
major = max(np.mean(y[test]), 1 - np.mean(y[test]))
print(f'\nB1. Inferability: {len(y)} decisions, {len(np.unique(eps))} episodes ({test.sum()} held out); '
      f'bit=1 share {np.mean(y):.1%}; always-majority {major:.1%}')
print(f'   held-out accuracy: MLP {acc_mlp:.2%}, lookup table {acc_tab:.2%}')
by_kind = {}
for k in np.unique(kinds[test]):
    m = kinds[test] == k
    by_kind[k] = dict(n=int(m.sum()), mlp=float((pred[m] == y[test][m]).mean()), bit1=float(y[test][m].mean()))
    print(f'     {k:13s} n={m.sum():5d} bit=1 {y[test][m].mean():.1%}  MLP {(pred[m] == y[test][m]).mean():.2%}')

# ── analysis B2
print('\nB2. Pointless decisions (every action, then straight at the next decision):')
for k, rows in B2_rows.items():
    n_diff = sum(bool(r['differ']) for r in rows)
    cnt = collections.Counter(a for r in rows for a in r['differ'])
    print(f'   {k:13s} {len(rows)} states: some action ends elsewhere than straight in {n_diff} '
          f'({100 * n_diff / max(1, len(rows)):.0f}%); by action {dict(cnt.most_common())}')

json.dump(dict(A=A_rows, A_yard=A_yard, A_detail=detail, B1=dict(n=len(y), acc_mlp=float(acc_mlp), acc_table=float(acc_tab),
               majority=float(major), by_kind=by_kind), B2={k: v for k, v in B2_rows.items()}), open(OUT, 'w'))
print(f'\nwrote {OUT} ({time.time() - t0:.0f}s)')
