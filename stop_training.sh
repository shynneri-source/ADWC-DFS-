#!/bin/bash
# filepath: /home/shyn/Dev/ADWC-DFS-/stop_training.sh

echo "🛑 ADWC-DFS Training Stopper"
echo "============================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/training_logs"
PID_FILE="$LOG_DIR/training.pid"

# Check if training is running
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No training process found."
    exit 1
fi

TRAINING_PID=$(cat "$PID_FILE")

if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "❌ Training process not running (PID: $TRAINING_PID)"
    echo "   Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 1
fi

echo "⚠️  Found training process (PID: $TRAINING_PID)"
echo ""

# Ask for confirmation
read -p "Are you sure you want to stop training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled."
    exit 0
fi

echo "🛑 Stopping training process..."

# Try graceful stop first
kill -TERM $TRAINING_PID 2>/dev/null

# Wait a few seconds
sleep 3

# Check if still running
if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "⚠️  Process still running, forcing stop..."
    kill -KILL $TRAINING_PID 2>/dev/null
    sleep 1
fi

# Final check
if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "❌ Failed to stop process!"
    exit 1
else
    echo "✅ Training stopped successfully."
fi

# Cleanup
rm -f "$PID_FILE"

# Show logs location
LATEST_LOG=$(ls -t "$LOG_DIR"/ensemble_training_*.log 2>/dev/null | head -1)
if [ ! -z "$LATEST_LOG" ]; then
    echo "📋 Training log: $LATEST_LOG"
fi

echo "🔄 To restart training: bash train_ensemble.sh"