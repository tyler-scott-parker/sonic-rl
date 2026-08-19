#!/usr/bin/env bash
# watchdog.sh — keep a multistage run alive across crashes.
#
# A 14-board run takes days. If the emulator hiccups or a worker dies at 3am you
# lose hours. This polls for the process and relaunches it (it resumes from the
# autosaved checkpoint, so a restart costs at most AUTOSAVE_STEPS).
#
# It will NOT restart when:
#   - you stopped it on purpose (STOP_<run> sentinel, or WATCHDOG_OFF)
#   - the log shows a GracefulStop (you asked it to stop)
#   - it's crash-looping (too many restarts too fast) -> bails loudly
#
# USAGE
#   chmod +x ~/sonic-rl/watchdog.sh
#   nohup ~/sonic-rl/watchdog.sh v5 > ~/sonic-rl/logs/watchdog_v5.log 2>&1 &
#
#   # stop the watchdog (does NOT stop training):
#   touch ~/sonic-rl/WATCHDOG_OFF
#
#   # stop BOTH cleanly:
#   touch ~/sonic-rl/WATCHDOG_OFF && touch ~/sonic-rl/STOP_v5
#
#   # see what it's been doing:
#   cat ~/sonic-rl/logs/watchdog_v5.log
#
# Was watchdog_v3.sh, hardcoded to one run. The run name is now the argument —
# the config drift that pattern caused is documented in train_ss_multistage.py.

set -u
cd "$HOME/sonic-rl" || exit 1

RUN="${1:-}"
if [[ -z "$RUN" ]]; then
  echo "usage: watchdog.sh <run>   (e.g. watchdog.sh v5; see train_ss_multistage.py --list)"
  exit 2
fi

TRAINER="train_ss_multistage.py"
TRAIN_LOG="logs/ss_multistage_${RUN}.log"
WD_OFF="$HOME/sonic-rl/WATCHDOG_OFF"
STOP_FILE="$HOME/sonic-rl/STOP_${RUN}"

POLL_SECS=120           # how often to check
MAX_RESTARTS=20         # total restarts before giving up
MIN_UPTIME_SECS=600     # a run that dies faster than this counts as a crash-loop
CRASHLOOP_LIMIT=3       # this many too-fast deaths in a row -> bail

restarts=0
fast_deaths=0

launch() {
  # The run name carries the whole config (stages, obs preset, hyperparameters,
  # log path) — no env vars to re-specify and get wrong. --resume because by
  # definition there is a checkpoint on disk by the time the watchdog relaunches.
  nohup python3 "$TRAINER" "$RUN" --resume >> "$TRAIN_LOG" 2>&1 &
  sleep 20   # give it time to boot / fail fast
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] watchdog started for run '$RUN' (poll ${POLL_SECS}s, max ${MAX_RESTARTS} restarts)"
echo "[$(stamp)] stop watchdog with: touch $WD_OFF"

while true; do
  if [[ -f "$WD_OFF" ]]; then
    echo "[$(stamp)] WATCHDOG_OFF present -> exiting (training left as-is)"
    rm -f "$WD_OFF"
    exit 0
  fi

  if pgrep -f "$TRAINER $RUN" > /dev/null; then
    fast_deaths=0            # healthy; reset the crash-loop counter
    sleep "$POLL_SECS"
    continue
  fi

  # --- process is gone: was it intentional? ---
  if [[ -f "$STOP_FILE" ]]; then
    echo "[$(stamp)] $STOP_FILE present -> intentional stop, not restarting"
    exit 0
  fi
  if tail -50 "$TRAIN_LOG" 2>/dev/null | grep -q "GracefulStop"; then
    echo "[$(stamp)] log shows GracefulStop -> intentional stop, not restarting"
    exit 0
  fi

  # --- unexpected death: decide whether to restart ---
  restarts=$((restarts + 1))
  if (( restarts > MAX_RESTARTS )); then
    echo "[$(stamp)] !! hit MAX_RESTARTS ($MAX_RESTARTS) -> giving up."
    echo "[$(stamp)]    Something is systematically wrong. Check $TRAIN_LOG."
    exit 1
  fi

  # crash-loop guard: how long did the last run survive?
  if [[ -f "$TRAIN_LOG" ]]; then
    log_age=$(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0) ))
  else
    log_age=99999
  fi
  if (( log_age < MIN_UPTIME_SECS )); then
    fast_deaths=$((fast_deaths + 1))
    echo "[$(stamp)] died after only ${log_age}s (fast death #$fast_deaths)"
    if (( fast_deaths >= CRASHLOOP_LIMIT )); then
      echo "[$(stamp)] !! CRASH LOOP: $CRASHLOOP_LIMIT fast deaths in a row -> giving up."
      echo "[$(stamp)]    Last 30 log lines:"
      tail -30 "$TRAIN_LOG" 2>/dev/null | sed 's/^/    | /'
      exit 1
    fi
    sleep 60   # brief backoff before retrying
  fi

  echo "[$(stamp)] trainer not running -> restart #$restarts"
  launch

  if pgrep -f "$TRAINER $RUN" > /dev/null; then
    resumed=$(grep -m1 -E "Resuming|FRESH" "$TRAIN_LOG" 2>/dev/null || echo "?")
    echo "[$(stamp)] relaunched OK — $resumed"
  else
    echo "[$(stamp)] relaunch FAILED to stay up. Last 20 log lines:"
    tail -20 "$TRAIN_LOG" 2>/dev/null | sed 's/^/    | /'
  fi

  sleep "$POLL_SECS"
done
