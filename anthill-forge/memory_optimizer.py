# memory_optimizer.py
import torch
import gc
import time
from threading import Thread

def continuous_memory_cleanup():
    """Background thread to keep memory tidy"""
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(60)  # Clean every minute

# Start in background
Thread(target=continuous_memory_cleanup, daemon=True).start()
