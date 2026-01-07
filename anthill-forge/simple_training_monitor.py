# simple_training_monitor.py
import time
from datetime import datetime
import psutil
import torch
import os

def find_training_process():
    """Find the Python training process"""
    training_keywords = ['train', 'instruction', 'anthill', 'forge']
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(keyword in cmdline.lower() for keyword in training_keywords):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None

def monitor_training():
    """Simple monitoring without log files"""
    print("\n" + "="*60)
    print("ANTHILL FORGE - SIMPLE MONITOR")
    print("="*60)
    
    # Find training process
    proc = find_training_process()
    if not proc:
        print("❌ No training process found!")
        print("Make sure train_instruction_model_fixed.py is running.")
        return
    
    print(f"✅ Found training process: PID {proc.pid}")
    print(f"   Started: {datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = proc.create_time()
    last_update = time.time()
    estimated_total_steps = 1815
    estimated_time_per_step = 45  # seconds, from your observation
    
    print("\n📊 Estimated Progress (based on time elapsed):")
    print("   This is an ESTIMATE since logs aren't available yet")
    print("="*60)
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            elapsed_hours = elapsed / 3600
            
            # Estimate progress based on time
            estimated_steps_completed = int(elapsed / estimated_time_per_step)
            if estimated_steps_completed > estimated_total_steps:
                estimated_steps_completed = estimated_total_steps
            
            progress_percent = (estimated_steps_completed / estimated_total_steps) * 100
            remaining_steps = estimated_total_steps - estimated_steps_completed
            eta_seconds = remaining_steps * estimated_time_per_step
            eta_hours = eta_seconds / 3600
            
            # Get system stats
            cpu_percent = proc.cpu_percent(interval=1)
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            
            # Clear and display
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "="*60)
            print("ANTHILL FORGE - SIMPLE MONITOR")
            print("="*60)
            print(f"\n📊 ESTIMATED PROGRESS (logs not available)")
            print(f"   Elapsed time: {elapsed_hours:.1f} hours")
            print(f"   Estimated steps: {estimated_steps_completed:,} / {estimated_total_steps:,}")
            print(f"   Estimated progress: {progress_percent:.1f}%")
            print(f"   Estimated ETA: {eta_hours:.1f} hours")
            
            print(f"\n💻 PROCESS STATS")
            print(f"   PID: {proc.pid}")
            print(f"   CPU: {cpu_percent:.1f}%")
            print(f"   Memory: {memory_mb:.0f} MB")
            
            # GPU stats if available
            if torch.cuda.is_available():
                gpu_mem_alloc = torch.cuda.memory_allocated() / 1e9
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"   GPU Memory: {gpu_mem_alloc:.2f}GB / {gpu_mem_reserved:.2f}GB")
            
            print(f"\n⏰ Last update: {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            print("Note: This is an ESTIMATE. Actual progress may vary.")
            print("Logs will appear after 10 steps (logging_steps=10)")
            print("Press Ctrl+C to exit")
            
            time.sleep(30)  # Update every 30 seconds
            
            # Refresh process handle
            try:
                proc = psutil.Process(proc.pid)
            except:
                print("\n⚠️ Training process ended!")
                break
                
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Training continues in background.")

if __name__ == "__main__":
    monitor_training()
