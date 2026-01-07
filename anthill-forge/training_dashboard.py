# training_dashboard_fixed.py
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import torch
import os
import sys

class TrainingDashboard:
    def __init__(self, log_dir="C:/anthill_forge_output"):
        self.log_dir = Path(log_dir)
        self.start_time = datetime.now()
        self.steps_completed = 0
        self.total_steps = 1815  # From your calculation
        self.step_times = []
        self.current_loss = 0.0
        
    def read_latest_logs(self):
        """Read and parse training logs"""
        log_file = self.log_dir / "logs" / "trainer_logs.json"
        if not log_file.exists():
            # Try to find any JSON log file
            log_files = list((self.log_dir / "logs").glob("*.json"))
            if log_files:
                log_file = log_files[0]
            else:
                return []
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except:
            return []
        
        logs = []
        for line in lines:
            try:
                logs.append(json.loads(line))
            except:
                continue
        return logs
    
    def get_system_stats(self):
        """Get CPU, GPU, memory stats"""
        stats = {}
        
        # CPU
        try:
            stats['cpu_percent'] = psutil.cpu_percent(interval=1)
            stats['cpu_cores'] = psutil.cpu_count(logical=True)
        except:
            stats['cpu_percent'] = 0
            stats['cpu_cores'] = 0
        
        # Memory
        try:
            mem = psutil.virtual_memory()
            stats['ram_used_gb'] = mem.used / (1024**3)
            stats['ram_total_gb'] = mem.total / (1024**3)
            stats['ram_percent'] = mem.percent
        except:
            stats['ram_used_gb'] = 0
            stats['ram_total_gb'] = 0
            stats['ram_percent'] = 0
        
        # GPU
        if torch.cuda.is_available():
            try:
                stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                stats['gpu_utilization'] = self.get_gpu_utilization()
            except:
                stats['gpu_memory_allocated_gb'] = 0
                stats['gpu_memory_reserved_gb'] = 0
                stats['gpu_utilization'] = 0
        
        return stats
    
    def get_gpu_utilization(self):
        """Get GPU utilization (Windows specific)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, shell=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().split('\n')[0])
        except:
            pass
        return 0
    
    def get_training_progress(self):
        """Get progress from training output if logs aren't being written"""
        # Alternative: read from stdout/stderr or check for any progress indicators
        # For now, return the last known values
        return self.steps_completed, self.current_loss
    
    def display_dashboard(self):
        """Display real-time dashboard"""
        print("\n" * 3)
        print("=" * 80)
        print("ANTHILL FORGE - TRAINING DASHBOARD")
        print("=" * 80)
        print("Initializing...")
        
        # Initial delay to allow logs to be created
        time.sleep(5)
        
        while True:
            # Clear screen (cross-platform)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Read logs
            logs = self.read_latest_logs()
            
            # Update from logs
            if logs:
                latest = logs[-1]
                self.steps_completed = latest.get('step', self.steps_completed)
                self.current_loss = latest.get('loss', self.current_loss)
                
                # Store step times for average calculation
                if 'train_runtime' in latest and self.steps_completed > 0:
                    avg_time = latest['train_runtime'] / self.steps_completed
                else:
                    avg_time = 0
            else:
                avg_time = 0
            
            # Calculate progress
            progress = (self.steps_completed / self.total_steps) * 100 if self.total_steps > 0 else 0
            elapsed = datetime.now() - self.start_time
            
            # Calculate ETA
            if self.steps_completed > 10 and avg_time > 0:
                remaining_steps = self.total_steps - self.steps_completed
                eta_seconds = remaining_steps * avg_time
                eta = timedelta(seconds=int(eta_seconds))
                completion_time = datetime.now() + eta
                completion_str = completion_time.strftime('%Y-%m-%d %H:%M:%S')
                eta_str = str(eta).split('.')[0]  # Remove microseconds
            else:
                eta_str = "Calculating..."
                completion_str = "Calculating..."
            
            # Get system stats
            stats = self.get_system_stats()
            
            # Display dashboard
            print("=" * 80)
            print("ANTHILL FORGE - TRAINING DASHBOARD")
            print("=" * 80)
            
            print(f"\n📊 TRAINING PROGRESS")
            print(f"   Steps: {self.steps_completed:,} / {self.total_steps:,} ({progress:.1f}%)")
            print(f"   Current Loss: {self.current_loss:.4f}")
            if avg_time > 0:
                print(f"   Avg Time/Step: {avg_time:.1f}s")
            else:
                print(f"   Avg Time/Step: Calculating...")
            print(f"   Elapsed: {str(elapsed).split('.')[0]}")
            print(f"   ETA: {eta_str}")
            print(f"   Completion: {completion_str}")
            
            print(f"\n💻 SYSTEM RESOURCES")
            print(f"   CPU: {stats.get('cpu_percent', 0):.1f}% ({stats.get('cpu_cores', 0)} cores)")
            print(f"   RAM: {stats.get('ram_used_gb', 0):.1f}/{stats.get('ram_total_gb', 0):.1f} GB ({stats.get('ram_percent', 0):.1f}%)")
            
            if torch.cuda.is_available():
                print(f"   GPU Memory: {stats.get('gpu_memory_allocated_gb', 0):.1f}/{stats.get('gpu_memory_reserved_gb', 0):.1f} GB")
                print(f"   GPU Utilization: {stats.get('gpu_utilization', 0):.1f}%")
            
            print(f"\n📁 TRAINING INFO")
            print(f"   Script: train_instruction_model_fixed.py")
            print(f"   Model: 10.6B parameters over network")
            print(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Logs: {self.log_dir / 'logs'}")
            
            print(f"\n" + "=" * 80)
            print("Press Ctrl+C to exit | Updates every 10 seconds")
            
            # Update every 10 seconds
            time.sleep(10)

if __name__ == "__main__":
    print("Starting Anthill Forge Dashboard...")
    
    # Try to auto-detect log directory
    possible_dirs = [
        "C:/anthill_forge_output",
        "X:/Anthill/anthill/anthill-forge",
        Path(__file__).parent
    ]
    
    log_dir = None
    for dir_path in possible_dirs:
        path = Path(dir_path)
        if path.exists():
            log_dir = path
            break
    
    if log_dir is None:
        log_dir = Path("C:/anthill_forge_output")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using log directory: {log_dir}")
    
    dashboard = TrainingDashboard(log_dir)
    try:
        dashboard.display_dashboard()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped. Training continues in background.")
        print(f"Training started at: {dashboard.start_time}")
        if dashboard.steps_completed > 0:
            print(f"Progress: {dashboard.steps_completed} steps completed")
    except Exception as e:
        print(f"\nDashboard error: {e}")
        print("Training continues in background.")
        input("Press Enter to close...")