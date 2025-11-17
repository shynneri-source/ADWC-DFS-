#!/bin/bash
# filepath: /home/shyn/Dev/ADWC-DFS-/train_ensemble.sh

echo "🚀 ADWC-DFS Ensemble Training Script"
echo "======================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/training_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/ensemble_training_$TIMESTAMP.log"
PID_FILE="$LOG_DIR/training.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print and log
log_and_print() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        log_and_print "❌ Training already running with PID: $OLD_PID"
        log_and_print "   Check: tail -f $LOG_DIR/ensemble_training_*.log"
        exit 1
    else
        log_and_print "🧹 Cleaning up old PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Parse arguments
N_MODELS=5
SAMPLE_FRAC=""
PYTHON_CMD="python"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--models)
            N_MODELS="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE_FRAC="$2"
            shift 2
            ;;
        --uv)
            PYTHON_CMD="uv run"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --models N     Number of models (default: 5)"
            echo "  -s, --sample F     Sample fraction (e.g., 0.1)"
            echo "  --uv               Use 'uv run' instead of 'python'"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                 # Train 5 models with full data"
            echo "  $0 -n 7            # Train 7 models"
            echo "  $0 -s 0.1          # Quick test with 10% data"
            echo "  $0 -n 3 -s 0.1     # 3 models, 10% data"
            echo "  $0 --uv -n 5       # Use uv run"
            exit 0
            ;;
        *)
            log_and_print "❌ Unknown option: $1"
            log_and_print "   Use -h for help"
            exit 1
            ;;
    esac
done

# Build command
CMD="$PYTHON_CMD ensemble_voting.py --n_models $N_MODELS"
if [ ! -z "$SAMPLE_FRAC" ]; then
    CMD="$CMD --sample_frac $SAMPLE_FRAC"
fi

log_and_print "🎯 Starting ADWC-DFS Ensemble Training"
log_and_print "======================================"
log_and_print "Timestamp: $(date)"
log_and_print "Command: $CMD"
log_and_print "Log file: $LOG_FILE"
log_and_print "PID file: $PID_FILE"
log_and_print ""

# Start training in background
log_and_print "🚀 Launching training process..."
cd "$SCRIPT_DIR"

# Start background process with proper function definition
nohup bash -c "
    # Function to run training
    run_training() {
        local log_file=\"\$1\"
        local cmd=\"\$2\"

        # Log system info
        {
            echo \"==================================\"
            echo \"SYSTEM INFORMATION\"
            echo \"==================================\"
            echo \"Date: \$(date)\"
            echo \"PWD: \$(pwd)\"
            echo \"Python: \$(which python)\"
            echo \"UV: \$(which uv)\"
            echo \"Command: \$cmd\"
            echo \"==================================\"
            echo \"\"
        } >> \"\$log_file\"

        # Run the actual training
        eval \"\$cmd\" >> \"\$log_file\" 2>&1
        local exit_code=\$?

        # Log completion
        {
            echo \"\"
            echo \"==================================\"
            echo \"TRAINING COMPLETED\"
            echo \"==================================\"
            echo \"Exit code: \$exit_code\"
            echo \"End time: \$(date)\"
            echo \"Log file: \$log_file\"
            echo \"==================================\"
        } >> \"\$log_file\"

        # If training was successful, run inference tests
        if [ \$exit_code -eq 0 ]; then
            echo \"\"
            echo \"==================================\"
            echo \"RUNNING INFERENCE TESTS\"
            echo \"==================================\"
            echo \"Starting inference tests on \$(date)\"
            echo \"\"

            # Run inference tests with default parameters
            INFER_CMD=\"$PYTHON_CMD inference_test.py --model_path results/ensemble_model.pkl --test_path data/test.csv --sample_size 1000 --num_samples 5\"
            echo \"Running: \$INFER_CMD\"

            eval \"\$INFER_CMD\" >> \"\$log_file\" 2>&1
            local infer_exit_code=\$?

            if [ \$infer_exit_code -eq 0 ]; then
                echo \"\"
                echo \"==================================\"
                echo \"INFERENCE TESTS COMPLETED SUCCESSFULLY\"
                echo \"==================================\"
                echo \"Inference tests completed on \$(date)\"
                echo \"Results saved to results/inference_tests/\"
                echo \"==================================\"
            else
                echo \"\"
                echo \"==================================\"
                echo \"INFERENCE TESTS FAILED\"
                echo \"==================================\"
                echo \"Inference tests failed on \$(date) with exit code: \$infer_exit_code\"
                echo \"==================================\"
            fi
        else
            echo \"\"
            echo \"==================================\"
            echo \"TRAINING FAILED - SKIPPING INFERENCE TESTS\"
            echo \"==================================\"
            echo \"Training failed with exit code: \$exit_code\"
            echo \"==================================\"
        fi

        return \$exit_code
    }

    # Call the function
    run_training '$LOG_FILE' '$CMD'
" > /dev/null 2>&1 &

TRAINING_PID=$!

# Save PID
echo $TRAINING_PID > "$PID_FILE"

log_and_print "✅ Training started successfully!"
log_and_print "   Process ID: $TRAINING_PID"
log_and_print "   Log file: $LOG_FILE"
log_and_print ""
log_and_print "📋 Monitoring commands:"
log_and_print "   Watch progress: tail -f $LOG_FILE"
log_and_print "   Check status:   bash monitor_training.sh"
log_and_print "   Stop training:  bash stop_training.sh"
log_and_print ""
log_and_print "🔥 You can now safely close this terminal!"
log_and_print "   Training will continue in background."
log_and_print "   After training completes, inference tests will run automatically."

# Wait a moment to ensure process started
sleep 2

# Check if process is still running
if ps -p $TRAINING_PID > /dev/null 2>&1; then
    log_and_print "✅ Training process confirmed running (PID: $TRAINING_PID)"
else
    log_and_print "❌ Training process failed to start!"
    log_and_print "   Check log file: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Training is now running in background."
echo "   After training completes, inference tests will run automatically."
echo "   Monitor: tail -f $LOG_FILE"