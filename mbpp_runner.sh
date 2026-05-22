#!/bin/bash
while true; do
    python run_zeroshot.py --benchmarks mbpp --n_attempts 3 >> zeroshot_mbpp.log 2>&1
    # Check if it completed (exit 0 and progress shows 500)
    TOTAL=$(python -c "import json; d=json.load(open('zeroshot_results/mbpp_progress.json')); print(d['total_so_far'])" 2>/dev/null)
    if [ "$TOTAL" = "500" ]; then
        echo "MBPP COMPLETE" >> zeroshot_mbpp.log
        break
    fi
    echo "$(date) Restarting MBPP from checkpoint..." >> zeroshot_mbpp.log
    sleep 5
done
