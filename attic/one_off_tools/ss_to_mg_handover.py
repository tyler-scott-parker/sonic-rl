#!/usr/bin/env python3
"""ss_to_mg_handover.py — run the special-stage run until its finish line, then give the box back to the
main-game run (Tyler 2026-10-15 ~22:30: "Would you be able to finish work on that final special stage and
then come back to the main game after it finishes?").

    setsid nohup python3 attic/one_off_tools/ss_to_mg_handover.py [SS_RUN] [MG_RUN] [BOARD] &   # d10 mg2 14

Finish line (sonic-rl-goal memory, 2026-10-12): SS_RUN's first fresh clear of BOARD, i.e. a line in
ss_log_<SS_RUN>.jsonl logged after this script started, on that stage, with no recording attached (a start
from the board's own start), success true. Then: STOP_<SS_RUN> (graceful save), the route-book harvest loop
stopped, MG_RUN resumed (python3 train_main.py MG_RUN --resume, as its supervisor expects), and after its
first steps the supervisor started (mg_supervise.sh MG_RUN 720). If SS_RUN exits before the finish line it is
resumed once; a second exit hands the box to MG_RUN anyway (it shouldn't sit idle). Every step is noted in
logs/overnight_20261015.log. Detached on purpose: it doesn't need the Claude session that started it.
"""
import datetime
import json
import os
import subprocess
import sys
import time

REPO = os.path.expanduser('~/Projects/sonic-rl')
os.chdir(REPO)
SS, MG = (sys.argv[1] if len(sys.argv) > 1 else 'd10'), (sys.argv[2] if len(sys.argv) > 2 else 'mg2')
BOARD = int(sys.argv[3]) if len(sys.argv) > 3 else 14
NOTE = 'logs/overnight_20261015.log'
SS_LOG = f'ss_log_{SS}.jsonl'
VENV = 'source venv/bin/activate && '


def note(msg):
    with open(NOTE, 'a') as f:
        f.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} HANDOVER: {msg}\n")


def pids(pattern):
    out = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def detach(cmd):
    """Start `cmd` (a bash line) in its own session, so it outlives this script and the shell."""
    subprocess.Popen(['bash', '-c', cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True, env={k: v for k, v in os.environ.items()
                                                                              if k != 'CUDA_VISIBLE_DEVICES'})


def wait_gone(pattern, limit):
    end = time.time() + limit
    while pids(pattern) and time.time() < end:
        time.sleep(5)
    return not pids(pattern)


def main():
    off = os.path.getsize(SS_LOG)                   # only what the run logs from now on
    restarts = 0
    note(f'watching {SS} for its first fresh clear of board {BOARD} (from byte {off:,} of {SS_LOG})')
    found = None
    while found is None:
        with open(SS_LOG, 'rb') as f:
            f.seek(off)
            data = f.read()
        cut = data.rfind(b'\n') + 1                     # whole lines only
        off += cut
        for line in data[:cut].splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if int(r.get('stage', 0)) == BOARD and not r.get('demo') and r.get('success'):
                found = r
                break
        if found is not None:
            break
        if not pids(f'^python3 train_ss_multistage.py {SS}'):
            if restarts == 0:
                restarts += 1
                note(f'{SS} exited before the finish line: resuming it once')
                with open(f'logs/ss_multistage_{SS}.log', 'a') as f:
                    f.write(f"=== resumed {datetime.datetime.now():%Y-%m-%d %H:%M} by ss_to_mg_handover.py after an exit ===\n")
                detach(VENV + f'exec python3 train_ss_multistage.py {SS} --resume >> logs/ss_multistage_{SS}.log 2>&1')
                time.sleep(120)
                continue
            note(f'{SS} exited again before the finish line: handing the box to {MG} anyway')
            break
        time.sleep(60)
    if found is not None:
        note(f"FINISH LINE: {SS}'s first fresh clear of board {BOARD}: {found['spheres']}/{found.get('stage_total')} "
             f"spheres in {found['steps']} steps (log time {found.get('t')})")
        open(f'STOP_{SS}', 'w').close()
        if not wait_gone(f'^python3 train_ss_multistage.py {SS}', 900):
            for p in pids(f'^python3 train_ss_multistage.py {SS}'):
                os.kill(p, 15)
            wait_gone(f'^python3 train_ss_multistage.py {SS}', 300)
        note(f'{SS} stopped: ' + subprocess.run(['bash', '-c', f'grep -E "GracefulStop|FinalSave" logs/ss_multistage_{SS}.log | tail -1'],
                                                  capture_output=True, text=True).stdout.strip()[:120])
    for p in pids('^python3 -u route_book.py harvest'):
        os.kill(p, 15)
    note('route-book harvest loop stopped')
    if pids(f'^python3 train_main.py {MG}'):
        note(f'{MG} already running: nothing to start')
        return
    with open(f'logs/main_{MG}.log', 'a') as f:
        f.write(f"=== resumed {datetime.datetime.now():%Y-%m-%d %H:%M} by ss_to_mg_handover.py after {SS}'s finish line ===\n")
    size = os.path.getsize(f'logs/main_{MG}.log')
    detach(VENV + f'exec python3 train_main.py {MG} --resume >> logs/main_{MG}.log 2>&1')
    note(f'{MG} resumed; waiting for its first steps')
    end = time.time() + 1800
    while time.time() < end:
        with open(f'logs/main_{MG}.log', 'rb') as f:
            f.seek(size)
            tail = f.read().decode(errors='replace')
        if 'total_timesteps' in tail:
            break
        if 'Traceback' in tail or not pids(f'^python3 train_main.py {MG}'):
            note(f'{MG} did NOT start: ' + tail[-300:].replace('\n', ' | '))
            return
        time.sleep(15)
    detach(f'exec attic/one_off_tools/mg_supervise.sh {MG} 720 > logs/mg_supervise_{MG}_{datetime.datetime.now():%H%M}.out 2>&1')
    note(f'{MG} stepping; supervisor started. Handover done.')


if __name__ == '__main__':
    main()
