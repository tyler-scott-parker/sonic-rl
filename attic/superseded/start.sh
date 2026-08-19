#!/bin/bash
cd ~/sonic-rl
source venv/bin/activate

echo "Killing any old trainers..."
pkill -9 -f train.py 2>/dev/null
sleep 1

echo "Clearing pycache..."
rm -rf __pycache__

echo "Clearing checkpoints..."
rm -f checkpoints/*.zip

echo "Starting trainer..."
python train.py &
TRAINER_PID=$!
echo "Trainer PID: $TRAINER_PID"

trap "echo 'Stopping...'; kill $TRAINER_PID 2>/dev/null; exit" SIGINT SIGTERM

sleep 3
echo "Starting watcher..."
python watch_checkpoint.py
kill $TRAINER_PID 2>/dev/null
