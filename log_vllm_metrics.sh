#!/bin/bash

# Set your time limit in seconds (e.g., 3600 = 1 hour)
TIME_LIMIT=$3600
OUTPUT_FILE="metrics_$(date +%Y%m%d_%H%M).log"
END_TIME=$((SECONDS + TIME_LIMIT))

echo "Logging vLLM metrics to $OUTPUT_FILE for $TIME_LIMIT seconds..."
echo "Log started at $(date)" >> "$OUTPUT_FILE"

while [ $SECONDS -lt $END_TIME ]; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    if curl -s http://0.0.0.0:8000/metrics > /tmp/metrics_tmp; then
        echo "=== $TIMESTAMP ===" >> "$OUTPUT_FILE"
        cat /tmp/metrics_tmp >> "$OUTPUT_FILE"
    else
        echo "=== $TIMESTAMP (ERROR: Could not fetch metrics) ===" >> "$OUTPUT_FILE"
    fi
    sleep 60
done

echo "Log ended at $(date)" >> "$OUTPUT_FILE"
echo "Done. Log saved to $OUTPUT_FILE"
