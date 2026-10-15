#!/bin/bash
# mg_supervise.sh RUN [MINUTES] — watch a main-game run like mg_watch.sh, and act on the
# stepping-stones flag by itself: mg_stones.py adds stones where a start point stalled, then the
# run is stopped and resumed so the env builds them (~7 min: the launch re-checks the recordings),
# and it exits to report that. Also prints and exits (so whoever started it hears about it) on any
# other ALERT or NEWS, a crash, a stall too fine for stones, or after MINUTES (default 720).
# Only ever stops and restarts a run that is running: a run paused for Tyler's gaming is left alone. Status every 10 min into logs/<run>_watch.log.
# Tyler 2026-10-12: "This game is riddled with precise jumps across all many of its stages."
RUN=${1:?usage: mg_supervise.sh RUN [MINUTES]}
cd ~/Projects/sonic-rl || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=
end=$(( $(date +%s) + ${2:-720} * 60 ))
running() { pgrep -f "^python3 train_main.py $RUN" >/dev/null; }
while true; do
  if ! running; then echo "$RUN NOT RUNNING ($(date +%T))"; tail -5 logs/main_$RUN.log; exit 0; fi
  out=$(nice -n 10 python3 attic/one_off_tools/mg_status.py $RUN --state logs/${RUN}_watch_state.json 2>&1)
  { echo "=== $(date '+%F %T')"; echo "$out"; } >> logs/${RUN}_watch.log
  if echo "$out" | grep -q "ALERT stepping stones"; then
    # stop first: the trainer rewrites the curriculum file when a pointer moves, which would
    # undo mg_stones.py's pointer change if it happened while the run was still going
    touch STOP_main_$RUN
    while running; do sleep 5; done
    stones=$(nice -n 10 python3 mg_stones.py $RUN 2>&1 | grep -v -i "warn\|pkg_res")
    { echo "=== $(date '+%F %T') mg_stones"; echo "$stones"; } >> logs/${RUN}_watch.log
    # the trainer gets the GPU back: the export above is for the tools (until 2026-10-15 it
    # leaked into this restart, and device='auto' would have resumed the run on the CPU)
    env -u CUDA_VISIBLE_DEVICES nohup python3 train_main.py $RUN --resume >> logs/main_$RUN.log 2>&1 &
    if echo "$stones" | grep -q "^wrote"; then
      echo "$(date +%T) stepping stones added, $RUN restarted (re-arm this, and a window following"
      echo "the training start point needs a restart: the start points were renumbered):"; echo "$stones"
    else
      echo "$(date +%T) stepping-stones flag, but no stones added ($RUN restarted unchanged):"; echo "$stones"; echo "$out"
    fi
    exit 0
  fi
  if echo "$out" | grep -q "ALERT\|NEWS"; then echo "$out"; exit 0; fi
  [ $(date +%s) -ge $end ] && { echo "$out"; exit 0; }
  sleep 600
done
