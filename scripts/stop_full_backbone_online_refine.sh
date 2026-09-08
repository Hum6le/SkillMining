#!/usr/bin/env bash

# Stop one full backbone online-refinement launch and all of its descendants.
# This operates only on the explicitly supplied PID; it never matches command
# text and therefore does not affect other experiments.

set -u

usage() {
    cat <<'EOF'
Usage: bash scripts/stop_full_backbone_online_refine.sh (--pid PID | --pid-file FILE)

Send SIGTERM to the exact full-online-refine launcher and its descendants.
After a short grace period, any remaining process in that same tree receives
SIGKILL. The PID file is the one printed by launch_full_backbone_online_refine.sh.
EOF
}

PID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)
            [[ $# -ge 2 ]] || { echo "Missing value for --pid" >&2; exit 2; }
            PID="$2"
            shift 2
            ;;
        --pid-file)
            [[ $# -ge 2 && -f "$2" ]] || {
                echo "PID file not found: ${2:-}" >&2
                exit 2
            }
            PID="$(tr -d '[:space:]' < "$2")"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$PID" =~ ^[0-9]+$ ]] || {
    echo "A numeric --pid or --pid-file is required." >&2
    exit 2
}
kill -0 "$PID" 2>/dev/null || {
    echo "Online-refine launcher is not running: $PID" >&2
    exit 1
}

descendants() {
    local parent_pid="$1"
    local child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        descendants "$child_pid"
        printf '%s\n' "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

send_signal_tree() {
    local signal_name="$1"
    local child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        kill "-$signal_name" "$child_pid" 2>/dev/null || true
    done < <(descendants "$PID")
    kill "-$signal_name" "$PID" 2>/dev/null || true
}

echo "Stopping full backbone online-refinement PID $PID and descendants..."
send_signal_tree TERM

for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PID" 2>/dev/null || {
        echo "Stopped cleanly."
        exit 0
    }
    sleep 1
done

echo "Grace period expired; force-stopping remaining descendants..." >&2
send_signal_tree KILL
echo "Stop signals sent. Verify with: ps -o pid,ppid,stat,cmd --forest -p $PID"
