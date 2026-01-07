# progress_monitor.py
import json
import time
from datetime import datetime
from pathlib import Path

log_file = Path("C:/anthill_forge_output/logs/trainer_logs.json")
progress_log = Path("C:/anthill_forge_output/training_progress.csv")

def monitor_training():
    last_size = 0
    steps = 0
    
    if progress_log.exists():
        with open(progress_log, 'r') as f:
            lines = f.readlines()
            if lines:
                steps = len(lines) - 1  # Subtract header
    
    while True:
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = lines[last_size:]
            last_size = len(lines)
            
            for line in new_lines:
                try:
                    entry = json.loads(line)
                    if 'loss' in entry:
                        step = entry.get('step', 0)
                        loss = entry['loss']
                        
                        # Append to progress CSV
                        with open(progress_log, 'a') as f:
                            f.write(f"{datetime.now()},{step},{loss}\n")
                        
                        # Calculate ETA
                        if steps > 10:
                            avg_time = (time.time() - start_time) / steps
                            remaining = (1815 - steps) * avg_time
                            hours = remaining / 3600
                            print(f"Step {step}: Loss={loss:.4f}, ETA={hours:.1f}h")
                except:
                    pass
        
        time.sleep(10)
