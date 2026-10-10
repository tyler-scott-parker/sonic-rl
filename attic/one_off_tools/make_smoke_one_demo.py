"""Writes smoke_one_demo.py next to this file: smoke_corners.py on d4's setup
for ONE recording, env version from CHK_ENV (default d4's v5). Regenerate after
any smoke_corners.py change.
  python3 attic/one_off_tools/make_smoke_one_demo.py
  CHK_ENV=7 python3 attic/one_off_tools/smoke_one_demo.py 6 stage6_20261010-092841"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
src = open(os.path.join(REPO, 'smoke_corners.py')).read()
pairs = [("HOME = os.path.dirname(os.path.abspath(__file__))", f"HOME = {REPO!r}\nsys.path.insert(0, HOME)"),
         ("run = sys.argv[1] if len(sys.argv) > 1 else 'd3'", "STAGE, PAT = int(sys.argv[1]), sys.argv[2]\nrun = f'new{STAGE}'"),
         ("if run not in ns['RUNS']:", "if 'd4' not in ns['RUNS']:"),
         ("cfg = ns['RUNS'][run]", "cfg = dict(ns['RUNS']['d4'], stages=[STAGE], demos=[PAT])  # d4's setup, one recording\n"
          "cfg['obs'] = dict(cfg['obs'], SS_ENV_VERSION=os.environ.get('CHK_ENV', cfg['obs']['SS_ENV_VERSION']))")]
for o, n in pairs:
    assert src.count(o) == 1, o
    src = src.replace(o, n)
open(os.path.join(HERE, 'smoke_one_demo.py'), 'w').write(src)
print('wrote', os.path.join(HERE, 'smoke_one_demo.py'))
