#!/bin/bash
# Quick helper script to run ACM Distilled for Wind Turbines and FD Fan
# 
# Usage:
#   ./run_distilled_quick.sh                     # Last 30 days
#   ./run_distilled_quick.sh 7                   # Last 7 days
#   ./run_distilled_quick.sh 7 reports/          # Last 7 days, save to reports/

set -e

# Configuration
EQUIPMENT=("WIND_TURBINE_01" "WIND_TURBINE_02" "FD_FAN")

# Parse arguments
DAYS=${1:-30}  # Default: last 30 days
OUTPUT_DIR=${2:-""}  # Default: console output

# Calculate time range
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
START_TIME=$(date -u -d "$DAYS days ago" +"%Y-%m-%dT%H:%M:%S" 2>/dev/null || date -u -v-${DAYS}d +"%Y-%m-%dT%H:%M:%S")

echo "================================================================================================"
echo "ACM DISTILLED BATCH RUNNER"
echo "================================================================================================"
echo "Time Range: Last $DAYS days ($START_TIME to $END_TIME)"
echo "Equipment: ${EQUIPMENT[@]}"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output: Saving to $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    echo "Output: Console"
fi
echo "================================================================================================"
echo ""

# Run for each equipment
SUCCESS_COUNT=0
TOTAL_COUNT=${#EQUIPMENT[@]}

for EQUIP in "${EQUIPMENT[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Running ACM Distilled: $EQUIP"
    echo "--------------------------------------------------------------------------------"
    
    if [ -n "$OUTPUT_DIR" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_FILE="$OUTPUT_DIR/${EQUIP}_${TIMESTAMP}.txt"
        
        if python acm_distilled.py \
            --equip "$EQUIP" \
            --start-time "$START_TIME" \
            --end-time "$END_TIME" \
            --output "$OUTPUT_FILE"; then
            echo "✓ SUCCESS: $EQUIP (saved to $OUTPUT_FILE)"
            ((SUCCESS_COUNT++))
        else
            echo "✗ FAILED: $EQUIP"
        fi
    else
        if python acm_distilled.py \
            --equip "$EQUIP" \
            --start-time "$START_TIME" \
            --end-time "$END_TIME"; then
            echo "✓ SUCCESS: $EQUIP"
            ((SUCCESS_COUNT++))
        else
            echo "✗ FAILED: $EQUIP"
        fi
    fi
done

# Summary
echo ""
echo "================================================================================================"
echo "BATCH RUN COMPLETE"
echo "================================================================================================"
echo "Successful: $SUCCESS_COUNT / $TOTAL_COUNT"
echo "================================================================================================"

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    exit 0
else
    exit 1
fi
