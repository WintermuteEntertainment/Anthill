# monitor_training.py
import json
import time
from datetime import datetime
from pathlib import Path

def monitor_training(log_dir):
    log_dir = Path(log_dir)
    log_file = log_dir / "trainer_logs.json"
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Monitoring training logs in {log_dir}")
    print("Press Ctrl+C to stop.")
    
    last_size = 0
    start_time = None
    steps = 0
    
    try:
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
                            steps = entry.get('step', 0)
                            loss = entry['loss']
                            
                            if start_time is None:
                                start_time = entry.get('epoch', 0)  # Not ideal, but we don't have start time
                            
                            print(f"Step {steps}: loss = {loss:.4f}")
                    except json.JSONDecodeError:
                        pass
                
                # Estimate time remaining
                if steps > 0 and start_time is not None:
                    current_time = time.time()
                    # We don't have the start time in the logs, so we can't compute elapsed time accurately
                    # Alternatively, we can use the first log entry's time if available.
            
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    log_dir = "C:/anthill_forge_output/logs"
    monitor_training(log_dir)