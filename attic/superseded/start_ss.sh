#!/bin/bash
# start_ss.sh — Start dedicated special stage training

cd ~/sonic-rl
source venv/bin/activate

echo "Killing any old SS trainers/watchers..."
pkill -f train_ss.py 2>/dev/null
pkill -f watch_ss.py 2>/dev/null
sleep 2

# Verify clean
if pgrep -f train_ss.py > /dev/null; then
    echo "ERROR: train_ss.py still running. Try: pkill -9 -f train_ss.py"
    exit 1
fi

mkdir -p logs

echo "Starting SS trainer..."
nohup python3 train_ss.py > logs/ss_trainer.log 2>&1 &
SS_PID=$!
echo "SS Trainer PID: $SS_PID"
echo ""
echo "To watch training:  python3 watch_ss.py"
echo "To tail the log:    tail -f ~/sonic-rl/logs/ss_trainer.log"
