#!/bin/bash
# filepath: /home/shyn/Dev/ADWC-DFS-/monitor_training.sh

echo "📊 ADWC-DFS Training Monitor"
echo "=============================="

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
    echo "   Start new training: bash train_ensemble.sh"
    exit 1
fi

# Find latest log file
LATEST_LOG=$(ls -t "$LOG_DIR"/ensemble_training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found."
    exit 1
fi

echo "✅ Training is running!"
echo "   Process ID: $TRAINING_PID"
echo "   Log file: $LATEST_LOG"
echo "   Started: $(ps -p $TRAINING_PID -o lstart= 2>/dev/null)"
echo ""

# Show last few lines
echo "📋 Recent progress (last 20 lines):"
echo "======================================"
tail -20 "$LATEST_LOG"
echo ""
echo "======================================"
echo "💡 Commands:"
echo "   Watch live: tail -f $LATEST_LOG"
echo "   Stop:       bash stop_training.sh"
echo "   Full log:   cat $LATEST_LOG"

# Show training stages if available
if grep -q "Training Model" "$LATEST_LOG"; then
    echo ""
    echo "🎯 Training Progress:"
    echo "===================="
    grep "Training Model\|Done in\|Ensemble training complete" "$LATEST_LOG" | tail -10
fi

# Show any errors
if grep -q "Error\|Exception\|Traceback" "$LATEST_LOG"; then
    echo ""
    echo "⚠️  Detected Errors:"
    echo "=================="
    grep -i "error\|exception\|traceback" "$LATEST_LOG" | tail -5
fi

# Show memory usage
echo ""
echo "💾 Resource Usage:"
echo "=================="
ps -p $TRAINING_PID -o pid,ppid,pcpu,pmem,etime,cmd 2>/dev/null || echo "Cannot get process info"

# Show file sizes
echo ""
echo "📁 Log File Info:"
echo "================="
ls -lh "$LATEST_LOG" 2>/dev/null || echo "Cannot get log file info"