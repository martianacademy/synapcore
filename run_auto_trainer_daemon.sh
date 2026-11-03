#!/bin/bash
# ==============================================================================
# SynapCore Automated 5 GB/Day Background Downloader & 256-Expert Trainer CLI
# (`run_auto_trainer_daemon.sh`)
# ==============================================================================
# Usage:
#   ./run_auto_trainer_daemon.sh start [--daily-limit-gb 5.0]  # Start background daemon
#   ./run_auto_trainer_daemon.sh status                        # Show live quota & training status
#   ./run_auto_trainer_daemon.sh run-batch                     # Download & train 1 batch immediately
#   ./run_auto_trainer_daemon.sh stop                          # Stop background daemon cleanly
#   ./run_auto_trainer_daemon.sh logs                          # Tail background daemon logs
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/opt/homebrew/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

PID_FILE="$SCRIPT_DIR/olmoe_merkle_store/datasets/auto_trainer_daemon.pid"
LOG_FILE="$SCRIPT_DIR/olmoe_merkle_store/datasets/auto_trainer_daemon.log"
PLIST_SRC="$SCRIPT_DIR/com.synapcore.autotrain.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.synapcore.autotrain.plist"

CMD="${1:-status}"
shift || true

case "$CMD" in
    start)
        if [ -f "$PID_FILE" ]; then
            OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
                echo "✅ SynapCore 5 GB/Day Auto-Trainer Daemon is ALREADY RUNNING (PID: $OLD_PID)."
                "$PYTHON_BIN" native/auto_daily_trainer_daemon.py --mode status "$@"
                exit 0
            fi
        fi
        mkdir -p "$SCRIPT_DIR/olmoe_merkle_store/datasets"
        nohup "$PYTHON_BIN" native/auto_daily_trainer_daemon.py --mode daemon "$@" > /dev/null 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "$PID_FILE"
        sleep 1
        echo "🚀 Started SynapCore 5 GB/Day Auto-Trainer Background Daemon (PID: $NEW_PID)"
        echo "   • Log File   : $LOG_FILE"
        echo "   • PID File   : $PID_FILE"
        echo "   • Behavior   : Auto-downloads & trains when internet connects; auto-stops at 5.00 GiB/day."
        "$PYTHON_BIN" native/auto_daily_trainer_daemon.py --mode status "$@"
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            PID_VAL="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$PID_VAL" ] && kill -0 "$PID_VAL" 2>/dev/null; then
                kill -TERM "$PID_VAL" 2>/dev/null || true
                sleep 1
                echo "🛑 Stopped SynapCore Auto-Trainer Daemon (PID: $PID_VAL)."
            else
                echo "ℹ️  Daemon process $PID_VAL was not running."
            fi
            rm -f "$PID_FILE"
        else
            pkill -f "auto_daily_trainer_daemon.py --mode daemon" 2>/dev/null || true
            echo "🛑 Stopped any running auto_daily_trainer_daemon.py processes."
        fi
        ;;
    run-batch|once)
        "$PYTHON_BIN" native/auto_daily_trainer_daemon.py --mode once "$@"
        ;;
    status)
        if [ -f "$PID_FILE" ]; then
            PID_VAL="$(cat "$PID_FILE" 2>/dev/null || true)"
            if [ -n "$PID_VAL" ] && kill -0 "$PID_VAL" 2>/dev/null; then
                echo "🟢 Background Daemon Process: RUNNING (PID: $PID_VAL)"
            else
                echo "⚪ Background Daemon Process: STOPPED (last PID: $PID_VAL)"
            fi
        else
            echo "⚪ Background Daemon Process: NOT STARTED (run './run_auto_trainer_daemon.sh start')"
        fi
        "$PYTHON_BIN" native/auto_daily_trainer_daemon.py --mode status "$@"
        ;;
    logs)
        tail -n 40 "$LOG_FILE"
        ;;
    install-launchd)
        mkdir -p "$HOME/Library/LaunchAgents"
        cp "$PLIST_SRC" "$PLIST_DST"
        launchctl unload "$PLIST_DST" 2>/dev/null || true
        launchctl load -w "$PLIST_DST"
        echo "✅ Installed & loaded macOS LaunchAgent: $PLIST_DST"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|run-batch|logs|install-launchd} [--daily-limit-gb 5.0]"
        exit 1
        ;;
esac
