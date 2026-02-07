#!/bin/bash
# ==============================================================================
# Ralph Wiggum Loop - Global Localization Development
# ==============================================================================
# Usage: bash ralph/run_ralph.sh [--max-iterations N]
#
# This loop feeds PROMPT.md to Claude Code repeatedly until the task is done.
# Progress is tracked in ralph/PROGRESS.md and persists across iterations.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAX_ITERATIONS=50
ITERATION=0

for arg in "$@"; do
    case $arg in
        --max-iterations=*) MAX_ITERATIONS="${arg#*=}" ;;
        --max-iterations) shift; MAX_ITERATIONS="$1" ;;
    esac
done

cd "$PROJECT_DIR"

echo "============================================"
echo "  RALPH WIGGUM LOOP"
echo "  Global Localization Development"
echo "  Max iterations: $MAX_ITERATIONS"
echo "============================================"
echo ""

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "============================================"
    echo "  ITERATION $ITERATION / $MAX_ITERATIONS"
    echo "  $(date)"
    echo "============================================"
    echo ""

    # Check if task is complete
    if [ -f "$SCRIPT_DIR/COMPLETE.txt" ]; then
        echo "COMPLETE.txt found! Task is done."
        echo "Final iteration: $ITERATION"
        cat "$SCRIPT_DIR/COMPLETE.txt"
        exit 0
    fi

    # Feed prompt to Claude Code
    cat "$SCRIPT_DIR/PROMPT.md" | claude \
        --print \
        --dangerously-skip-permissions \
        --max-turns 30

    echo ""
    echo "--- Iteration $ITERATION complete ---"
    echo ""

    # Brief pause between iterations
    sleep 2
done

echo "Max iterations ($MAX_ITERATIONS) reached without completion."
exit 1
