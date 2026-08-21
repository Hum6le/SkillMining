#!/usr/bin/env bash

# Stop one full ABCD experiment coordinator and every process below it.

set -u

usage() {
    cat <<'EOF'
Usage: bash scripts/stop_full_abcd_experiments.sh (--pid PID | --pid-file FILE)

Send SIGTERM to the exact coordinator process and recursively terminate its
worker Bash/Python descendants. This never matches processes by command text.
EOF
}

PID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) PID="$2"; shift 2 ;;
        --pid-file)
            [[ -f "$2" ]] || { echo "PID file not found: $2" >&2; exit 2; }
            PID="$(tr -d '[:space:]' < "$2")"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$PID" =~ ^[0-9]+$ ]] || { echo "A numeric --pid or --pid-file is required." >&2; exit 2; }
kill -0 "$PID" 2>/dev/null || { echo "Coordinator PID is not running: $PID" >&2; exit 1; }

stop_descendants() {
    local parent_pid="$1"
    local child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        stop_descendants "$child_pid"
        kill -TERM "$child_pid" 2>/dev/null || true
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

echo "Stopping full ABCD experiment coordinator PID $PID and its descendants..."
stop_descendants "$PID"
kill -TERM "$PID" 2>/dev/null || true
echo "SIGTERM sent. Check remaining processes with: ps -o pid,ppid,stat,cmd --forest -p $PID"
