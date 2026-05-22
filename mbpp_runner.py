"""Auto-restart wrapper for MBPP evaluation that resumes from checkpoint."""
import subprocess
import json
import time
import os

PROGRESS_FILE = "zeroshot_results/mbpp_progress.json"
MAX_RESTARTS = 50

for attempt in range(MAX_RESTARTS):
    print(f"[Restart {attempt+1}] Starting MBPP run...")
    result = subprocess.run(
        ["python", "run_zeroshot.py", "--benchmarks", "mbpp", "--n_attempts", "3"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or "."
    )
    
    # Check progress
    try:
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        total = progress["total_so_far"]
        correct = progress["correct"]
        rate = correct / total * 100 if total > 0 else 0
        print(f"  Progress: {correct}/{total} = {rate:.1f}%")
        
        if total >= 500:
            print("MBPP COMPLETE!")
            break
    except Exception as e:
        print(f"  Could not read progress: {e}")
    
    print(f"  Process exited with code {result.returncode}, restarting in 5s...")
    time.sleep(5)
else:
    print(f"Gave up after {MAX_RESTARTS} restarts")
