#!/bin/bash
# filepath: /home/shyn/Dev/ADWC-DFS-/watch_training.sh

echo "👀 ADWC-DFS Training Watcher"
echo "============================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/training_logs"
PID_FILE="$LOG_DIR/training.pid"

# Check if training is running
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No training process found."
    echo "   Start training: bash train_ensemble.sh"
    exit 1
fi

TRAINING_PID=$(cat "$PID_FILE")

if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "❌ Training process not running (PID: $TRAINING_PID)"
    echo "   Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 1
fi

# Find latest log file
LATEST_LOG=$(ls -t "$LOG_DIR"/ensemble_training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found."
    echo "   Expected location: $LOG_DIR/ensemble_training_*.log"
    echo ""
    echo "💡 Available commands:"
    echo "   python view_logs.py --training_logs  # Check training logs"
    echo "   ls -la $LOG_DIR/                    # List log directory"
    exit 1
fi

echo "👀 Watching training progress..."
echo "   Process ID: $TRAINING_PID"
echo "   Log file: $LATEST_LOG"
echo "   Press Ctrl+C to stop watching (training continues)"
echo ""
echo "======================================"

# Watch the log file
tail -f "$LATEST_LOG"