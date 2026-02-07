#!/bin/bash
# ==============================================================================
# Global Localization Test Harness
# ==============================================================================
# Builds the project and runs replay_localization against all recorded sessions
# WITHOUT a hint (--no-hint). Evaluates success rate, timing, and accuracy.
#
# Usage: bash ralph/test_localization.sh [--verbose] [--max-time 60]
#
# Exit codes:
#   0 = All recordings localized successfully
#   1 = Some recordings failed
#   2 = Build failed
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
RECORDINGS_DIR="C:/Users/wmuld/OneDrive/Desktop/Documents/ATLASCpp"
MAP_FILE="$RECORDINGS_DIR/house.ply"
RESULTS_DIR="$SCRIPT_DIR/results"
REPLAY_EXE="$BUILD_DIR/Release/replay_localization.exe"

MAX_TIME=60
VERBOSE=""
for arg in "$@"; do
    case $arg in
        --verbose) VERBOSE="--verbose" ;;
        --max-time) shift; MAX_TIME="$2" ;;
        --max-time=*) MAX_TIME="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ==============================================================================
# Step 1: Build
# ==============================================================================
echo "============================================"
echo "  BUILDING PROJECT"
echo "============================================"

cd "$BUILD_DIR"
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" \
    replay_localization.vcxproj -p:Configuration=Release -p:Platform=x64 -v:minimal 2>&1

if [ $? -ne 0 ]; then
    echo "BUILD FAILED"
    exit 2
fi

if [ ! -f "$REPLAY_EXE" ]; then
    echo "ERROR: replay_localization.exe not found after build"
    exit 2
fi

echo "Build successful."
echo ""

# ==============================================================================
# Step 2: Run against all recordings
# ==============================================================================
echo "============================================"
echo "  RUNNING LOCALIZATION TESTS (no hint)"
echo "  Max time per test: ${MAX_TIME}s"
echo "============================================"
echo ""

TOTAL=0
PASSED=0
FAILED=0
TOTAL_TIME=0
BEST_TIME=999999
WORST_TIME=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/summary_${TIMESTAMP}.txt"

echo "Test run: $TIMESTAMP" > "$SUMMARY_FILE"
echo "Max time: ${MAX_TIME}s" >> "$SUMMARY_FILE"
echo "Mode: --no-hint (global search)" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for recording in "$RECORDINGS_DIR"/loc_session_*.bin; do
    if [ ! -f "$recording" ]; then
        continue
    fi

    BASENAME=$(basename "$recording")
    TOTAL=$((TOTAL + 1))
    RESULT_JSON="$RESULTS_DIR/${BASENAME%.bin}_result.json"

    echo "--- Test $TOTAL: $BASENAME ---"

    # Run replay with --no-hint and capture JSON output
    START_SEC=$(date +%s)
    "$REPLAY_EXE" \
        --input "$recording" \
        --map "$MAP_FILE" \
        --no-hint \
        --max-time "$MAX_TIME" \
        --json \
        $VERBOSE \
        > "$RESULT_JSON" 2>&1
    EXIT_CODE=$?
    END_SEC=$(date +%s)
    WALL_TIME=$((END_SEC - START_SEC))

    # Parse JSON result (simple grep-based for portability)
    STATUS=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$RESULT_JSON" | head -1 | sed 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    CONFIDENCE=$(grep -o '"confidence"[[:space:]]*:[[:space:]]*[0-9.]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')
    REPLAY_TIME=$(grep -o '"replay_time_s"[[:space:]]*:[[:space:]]*[0-9.]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')
    POSE_X=$(grep -o '"x"[[:space:]]*:[[:space:]]*[-0-9.]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')
    POSE_Y=$(grep -o '"y"[[:space:]]*:[[:space:]]*[-0-9.]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')
    POSE_YAW=$(grep -o '"yaw_deg"[[:space:]]*:[[:space:]]*[-0-9.]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')
    ATTEMPTS=$(grep -o '"attempts"[[:space:]]*:[[:space:]]*[0-9]*' "$RESULT_JSON" | head -1 | sed 's/.*:[[:space:]]*//')

    # Default values if parsing fails
    [ -z "$STATUS" ] && STATUS="PARSE_ERROR"
    [ -z "$CONFIDENCE" ] && CONFIDENCE="0"
    [ -z "$REPLAY_TIME" ] && REPLAY_TIME="$WALL_TIME"
    [ -z "$ATTEMPTS" ] && ATTEMPTS="?"

    if [ "$STATUS" = "SUCCESS" ]; then
        PASSED=$((PASSED + 1))
        RESULT_STR="PASS"
    else
        FAILED=$((FAILED + 1))
        RESULT_STR="FAIL"
    fi

    echo "  Result: $RESULT_STR | Status: $STATUS | Confidence: $CONFIDENCE"
    echo "  Time: ${REPLAY_TIME}s | Attempts: $ATTEMPTS | Exit: $EXIT_CODE"
    if [ -n "$POSE_X" ]; then
        echo "  Pose: ($POSE_X, $POSE_Y) yaw=${POSE_YAW}deg"
    fi
    echo ""

    # Append to summary
    echo "Recording: $BASENAME" >> "$SUMMARY_FILE"
    echo "  Result: $RESULT_STR | Status: $STATUS | Confidence: $CONFIDENCE" >> "$SUMMARY_FILE"
    echo "  Time: ${REPLAY_TIME}s | Attempts: $ATTEMPTS" >> "$SUMMARY_FILE"
    echo "  Pose: ($POSE_X, $POSE_Y) yaw=${POSE_YAW}deg" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

# ==============================================================================
# Step 3: Summary
# ==============================================================================
echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "Total: $TOTAL | Passed: $PASSED | Failed: $FAILED"
PASS_RATE=0
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASSED * 100 / TOTAL))
fi
echo "Pass rate: ${PASS_RATE}%"
echo ""

echo "" >> "$SUMMARY_FILE"
echo "SUMMARY: $PASSED/$TOTAL passed (${PASS_RATE}%)" >> "$SUMMARY_FILE"

echo "Results saved to: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
