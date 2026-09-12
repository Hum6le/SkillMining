#!/usr/bin/env bash

# Stop a Trace2Skill module-ablation launcher and all variant workers below it.
set -u

usage() {
    cat <<'EOF'
Usage: bash scripts/stop_trace2skill_module_ablation.sh (--pid PID | --pid-file FILE)

Send SIGTERM to the exact launcher and recursively to its worker descendants.
After a short grace period, remaining processes in that same tree receive
SIGKILL. No command-text matching is used.
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
    echo "Trace2Skill module-ablation launcher is not running: $PID" >&2
    exit 1
}

descendants() {
    local parent="$1" child
    while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        descendants "$child"
        printf '%s\n' "$child"
    done < <(pgrep -P "$parent" 2>/dev/null || true)
}

signal_tree() {
    local signal_name="$1" child
    while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        kill "-$signal_name" "$child" 2>/dev/null || true
    done < <(descendants "$PID")
    kill "-$signal_name" "$PID" 2>/dev/null || true
}

echo "Stopping Trace2Skill module-ablation PID $PID and descendants..."
signal_tree TERM

for _ in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Stopped cleanly."
        exit 0
    fi
    sleep 1
done

echo "Grace period expired; force-stopping remaining descendants..." >&2
signal_tree KILL
echo "Stop signals sent. Verify with: ps -o pid,ppid,stat,cmd --forest -p $PID"
