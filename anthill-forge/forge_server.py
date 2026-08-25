# forge_server.py — Local HTTP API server for Anthill Forge
#
# This lightweight server acts as a bridge between the Anthill Forge Chrome
# extension (the dashboard) and the local training/export Python scripts.
# It runs on localhost:7800 and provides a REST API for:
#   - Checking system status (GPU, RAM, CUDA)
#   - Listing available datasets
#   - Starting/stopping QLoRA training (runs train_qlora.py as a subprocess)
#   - Starting/stopping GGUF export (runs export_gguf.py as a subprocess)
#   - Reading live training progress from a status file
#
# Usage:
#   python forge_server.py              # start on default port 7800
#   python forge_server.py --port 8888  # custom port
#
# The server uses Python's built-in http.server — no Flask dependency needed.

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Try importing optional dependencies for system info ──────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Import shared config for paths ───────────────────────────────
try:
    import config_qlora as cfg
except ImportError:
    # If config not found, use sensible defaults
    class cfg:
        SCRIPT_DIR = Path(__file__).parent
        PROJECT_ROOT = SCRIPT_DIR.parent
        DATA_FOLDER = PROJECT_ROOT / "datasets" / "processed"
        OUTPUT_DIR = Path("C:/anthill_forge_output/qlora")
        MERGED_DIR = OUTPUT_DIR / "merged"


# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL STATE — tracks what the server is currently doing
# ═══════════════════════════════════════════════════════════════════════

# The server state — shared across request handlers via this dict.
# Only one job (training or export) runs at a time.
server_state = {
    "state": "idle",       # idle, downloading, training, exporting, done, error
    "error": None,         # error message if state is "error"
    "training": {},        # live training metrics (from status file)
    "export": {},          # live export progress
    "download": {},        # model download progress
    "done": {},            # results after completion
    "process": None,       # subprocess.Popen reference (not serialized)
    "process_type": None,  # "training", "export", or "download"
}
state_lock = threading.Lock()

# Path where train_qlora.py writes its live status updates
STATUS_FILE = cfg.OUTPUT_DIR / "training_status.json"


# ═══════════════════════════════════════════════════════════════════════
#  SYSTEM INFO — GPU, VRAM, RAM detection
# ═══════════════════════════════════════════════════════════════════════

def get_system_info():
    """Gather GPU and system info for the dashboard."""
    info = {
        "cuda_available": False,
        "gpu_name": None,
        "vram_total": None,
        "vram_used": None,
        "ram_total": None,
        "ram_used": None,
    }

    if HAS_TORCH and torch.cuda.is_available():
        info["cuda_available"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total"] = round(props.total_memory / 1e9, 1)
        info["vram_used"] = round(torch.cuda.memory_allocated(0) / 1e9, 1)

    if HAS_PSUTIL:
        ram = psutil.virtual_memory()
        info["ram_total"] = round(ram.total / 1e9, 1)
        info["ram_used"] = round(ram.used / 1e9, 1)

    return info


# ═══════════════════════════════════════════════════════════════════════
#  DATASET DISCOVERY — find available JSONL training files
# ═══════════════════════════════════════════════════════════════════════

def list_datasets():
    """Find all JSONL files in the processed data folder."""
    datasets = []
    data_folder = cfg.DATA_FOLDER

    if not data_folder.exists():
        return datasets

    # Look for both *_clean.jsonl and regular .jsonl files
    for f in sorted(data_folder.glob("*.jsonl")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
            datasets.append({
                "name": f.name,
                "path": str(f),
                "pairs": count,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })
        except Exception:
            pass

    return datasets


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING STATUS — read the status file written by train_qlora.py
# ═══════════════════════════════════════════════════════════════════════

def read_training_status():
    """
    Read the live training status from the JSON file that train_qlora.py
    writes during training. Returns a dict with current step, loss, etc.
    """
    if not STATUS_FILE.exists():
        return {}

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # File might be partially written — that's OK, just return empty
        return {}


# ═══════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD — download HuggingFace models before training
# ═══════════════════════════════════════════════════════════════════════

def check_model_cached(model_id):
    """Check if a HuggingFace model is already downloaded and cached."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                size_gb = round(repo.size_on_disk / 1e9, 1)
                return {"cached": True, "size_gb": size_gb, "model_id": model_id}
        return {"cached": False, "model_id": model_id}
    except ImportError:
        return {"cached": False, "model_id": model_id, "warning": "huggingface_hub not installed"}
    except Exception as e:
        return {"cached": False, "model_id": model_id, "warning": str(e)}


def start_model_download(model_id):
    """Launch huggingface-cli download as a subprocess."""
    with state_lock:
        if server_state["state"] not in ("idle", "done", "error"):
            raise RuntimeError(f"Cannot start download: server is {server_state['state']}")

    # Use an inline script for progress reporting
    download_script = f"""
import sys, os
from huggingface_hub import snapshot_download, HfApi

model_id = "{model_id}"
print(f"Resolving {{model_id}}...")
sys.stdout.flush()

try:
    api = HfApi()
    info = api.model_info(model_id)
    siblings = info.siblings or []
    total_files = len(siblings)
    total_bytes = sum(s.size or 0 for s in siblings)
    print(f"Fetching {{total_files}} files ({{total_bytes / 1e9:.1f}} GB)")
    sys.stdout.flush()
except Exception as e:
    print(f"Could not get model info: {{e}}")
    total_files = 0
    sys.stdout.flush()

print(f"Downloading {{model_id}}...")
sys.stdout.flush()

# Enable HF transfer progress via env
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

path = snapshot_download(model_id)
print(f"Download complete: {{path}}")
sys.stdout.flush()
"""
    cmd = [sys.executable, "-c", download_script]

    print(f"[Forge Server] Starting model download: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )

    with state_lock:
        server_state["state"] = "downloading"
        server_state["error"] = None
        server_state["process"] = proc
        server_state["process_type"] = "download"
        server_state["download"] = {
            "model_id": model_id,
            "phase": "Starting download...",
            "message": f"Downloading {model_id} from HuggingFace...",
            "files_done": 0,
            "files_total": 0,
        }

    thread = threading.Thread(
        target=_monitor_download, args=(proc, model_id), daemon=True
    )
    thread.start()
    return True


def _monitor_download(proc, model_id):
    """Background thread that watches the download subprocess."""
    output_lines = []
    files_done = 0

    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            output_lines.append(line)
            print(f"[download] {line}")

            # Parse progress from huggingface-cli output
            with state_lock:
                dl = server_state["download"]
                if "Fetching" in line and "files" in line:
                    # e.g. "Fetching 15 files: ..."
                    try:
                        parts = line.split()
                        idx = parts.index("Fetching") + 1
                        dl["files_total"] = int(parts[idx])
                    except (ValueError, IndexError):
                        pass
                elif "Downloading" in line or "downloading" in line:
                    dl["message"] = line.strip()[:120]
                elif line.strip().endswith(".safetensors") or line.strip().endswith(".json") or line.strip().endswith(".model"):
                    files_done += 1
                    dl["files_done"] = files_done
                    dl["message"] = f"Downloaded: {line.strip()[-80:]}"

                # Update phase based on activity
                if files_done > 0 and dl.get("files_total", 0) > 0:
                    dl["phase"] = f"Downloading files ({files_done}/{dl['files_total']})..."
                elif files_done > 0:
                    dl["phase"] = f"Downloading files ({files_done} so far)..."
    except Exception:
        pass

    returncode = proc.wait()

    with state_lock:
        server_state["process"] = None
        server_state["process_type"] = None

        if returncode == 0:
            server_state["state"] = "done"
            server_state["error"] = None
            server_state["done"] = {
                "type": "download",
                "message": f"Model {model_id} downloaded successfully! Ready to train.",
                "model_id": model_id,
            }
        else:
            server_state["state"] = "error"
            tail = output_lines[-5:] if output_lines else ["No output"]
            server_state["error"] = f"Download failed (code {returncode}).\n" + "\n".join(tail)

    print(f"[Forge Server] download finished with code {returncode}")


# ═══════════════════════════════════════════════════════════════════════
#  JOB MANAGEMENT — start, stop, and monitor subprocesses
# ═══════════════════════════════════════════════════════════════════════

def start_training(config):
    """
    Launch train_qlora.py as a subprocess with the given config.
    Returns True on success, raises on error.
    """
    with state_lock:
        if server_state["state"] not in ("idle", "done", "error"):
            raise RuntimeError(f"Cannot start training: server is {server_state['state']}")

    # Build CLI arguments from the config dict
    script = str(Path(__file__).parent / "train_qlora.py")
    cmd = [sys.executable, script]

    if config.get("model"):
        cmd.extend(["--model", config["model"]])
    if config.get("epochs"):
        cmd.extend(["--epochs", str(config["epochs"])])
    if config.get("batch_size"):
        cmd.extend(["--batch-size", str(config["batch_size"])])
    if config.get("grad_accum"):
        cmd.extend(["--grad-accum", str(config["grad_accum"])])
    if config.get("lr"):
        cmd.extend(["--lr", str(config["lr"])])
    if config.get("max_length"):
        cmd.extend(["--max-length", str(config["max_length"])])
    if config.get("lora_r"):
        cmd.extend(["--lora-r", str(config["lora_r"])])
    if config.get("lora_alpha"):
        cmd.extend(["--lora-alpha", str(config["lora_alpha"])])
    if config.get("lora_dropout"):
        cmd.extend(["--lora-dropout", str(config["lora_dropout"])])
    if config.get("max_hours"):
        cmd.extend(["--max-hours", str(config["max_hours"])])
    if config.get("quant_bits"):
        cmd.extend(["--quant-bits", str(config["quant_bits"])])
    if config.get("skip_merge"):
        cmd.append("--skip-merge")

    # Remove old status file so we don't show stale data
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()

    # Launch the training process
    print(f"[Forge Server] Starting training: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )

    with state_lock:
        server_state["state"] = "training"
        server_state["error"] = None
        server_state["process"] = proc
        server_state["process_type"] = "training"
        server_state["training"] = {
            "phase": "Starting...",
            "message": "Launching training subprocess...",
            "current_step": 0,
            "total_steps": 0,
        }

    # Monitor the process in a background thread
    thread = threading.Thread(target=_monitor_process, args=(proc, "training", config), daemon=True)
    thread.start()

    return True


def start_export(config):
    """Launch export_gguf.py as a subprocess."""
    with state_lock:
        if server_state["state"] not in ("idle", "done", "error"):
            raise RuntimeError(f"Cannot start export: server is {server_state['state']}")

    script = str(Path(__file__).parent / "export_gguf.py")
    cmd = [sys.executable, script]

    if config.get("quant_type"):
        cmd.extend(["--quant", config["quant_type"]])

    print(f"[Forge Server] Starting export: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )

    with state_lock:
        server_state["state"] = "exporting"
        server_state["error"] = None
        server_state["process"] = proc
        server_state["process_type"] = "export"
        server_state["export"] = {
            "phase": "Starting export...",
            "message": "Launching export subprocess...",
            "progress": 0,
        }

    thread = threading.Thread(target=_monitor_process, args=(proc, "export", config), daemon=True)
    thread.start()

    return True


def stop_current_job():
    """Stop whatever job is currently running by sending SIGTERM/CTRL_BREAK."""
    with state_lock:
        proc = server_state.get("process")
        if proc is None or proc.poll() is not None:
            return False

    print("[Forge Server] Stopping current job...")

    try:
        if os.name == 'nt':
            # On Windows, send CTRL_BREAK to the process group
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception as e:
        print(f"[Forge Server] Error stopping process: {e}")
        try:
            proc.kill()
        except Exception:
            pass

    return True


def _monitor_process(proc, job_type, config):
    """
    Background thread that watches a subprocess until it exits.
    Reads stdout for log output and updates server_state accordingly.
    When the process finishes, transitions state to 'done' or 'error'.
    """
    output_lines = []

    # Read output line by line (for logging and parsing)
    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            output_lines.append(line)
            print(f"[{job_type}] {line}")

            # For training, also read the status file for live metrics
            if job_type == "training":
                status = read_training_status()
                if status:
                    with state_lock:
                        server_state["training"] = status

            # For export, parse progress from stdout
            if job_type == "export":
                _parse_export_progress(line)
    except Exception:
        pass

    # Wait for process to finish
    returncode = proc.wait()

    # Determine final state
    with state_lock:
        server_state["process"] = None
        server_state["process_type"] = None

        if returncode == 0:
            server_state["state"] = "done"
            server_state["error"] = None

            if job_type == "training":
                # Read final training status
                final = read_training_status()
                server_state["done"] = {
                    "type": "training",
                    "message": "QLoRA training complete! Model merged and ready for GGUF export.",
                    "model": config.get("model", "unknown"),
                    "final_loss": final.get("loss"),
                    "elapsed_seconds": final.get("elapsed_seconds"),
                }
            else:
                server_state["done"] = {
                    "type": "export",
                    "message": "GGUF export complete! Model is ready for llama-server.",
                }

        elif returncode == -2 or returncode == 3221225786:
            # SIGINT / Ctrl+C — graceful stop, checkpoint saved
            server_state["state"] = "done"
            server_state["done"] = {
                "type": job_type,
                "message": f"{'Training' if job_type == 'training' else 'Export'} stopped. Checkpoint saved.",
            }

        else:
            server_state["state"] = "error"
            # Grab last few lines of output for the error message
            tail = output_lines[-5:] if output_lines else ["No output captured"]
            server_state["error"] = f"Process exited with code {returncode}. Last output:\n" + \
                "\n".join(tail)

    print(f"[Forge Server] {job_type} finished with code {returncode}")


def _parse_export_progress(line):
    """Parse export script stdout lines to estimate progress."""
    with state_lock:
        exp = server_state["export"]

        if "[1/5]" in line:
            exp["phase"] = "Validating model"
            exp["progress"] = 10
            exp["message"] = line.strip()
        elif "[2/5]" in line:
            exp["phase"] = "Preparing output"
            exp["progress"] = 20
            exp["message"] = line.strip()
        elif "[3/5]" in line:
            exp["phase"] = "Converting to F16 GGUF"
            exp["progress"] = 30
            exp["message"] = "Converting HF model to GGUF (this takes a while)..."
        elif "[4/5]" in line:
            exp["phase"] = "Quantizing"
            exp["progress"] = 70
            exp["message"] = line.strip()
        elif "[5/5]" in line:
            exp["phase"] = "Validating GGUF"
            exp["progress"] = 90
            exp["message"] = line.strip()
        elif "EXPORT COMPLETE" in line:
            exp["progress"] = 100
            exp["message"] = "Export complete!"


# ═══════════════════════════════════════════════════════════════════════
#  HTTP REQUEST HANDLER — routes API requests
# ═══════════════════════════════════════════════════════════════════════

class ForgeHandler(BaseHTTPRequestHandler):
    """
    Handles incoming HTTP requests from the Forge Chrome extension.
    All routes are under /api/ and return JSON responses.
    """

    def handle(self):
        """Suppress ConnectionAbortedError on Windows. This happens when the
        extension's fetch times out before the response finishes sending."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/api/status':
            self._handle_status()
        elif self.path == '/api/datasets':
            self._handle_datasets()
        elif self.path == '/api/config':
            self._handle_get_config()
        elif self.path.startswith('/api/model/check'):
            self._handle_model_check()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/training/start':
            self._handle_start_training()
        elif self.path == '/api/training/stop':
            self._handle_stop()
        elif self.path == '/api/export/start':
            self._handle_start_export()
        elif self.path == '/api/export/stop':
            self._handle_stop()
        elif self.path == '/api/model/download':
            self._handle_model_download()
        elif self.path == '/api/model/stop':
            self._handle_stop()
        elif self.path == '/api/reset':
            self._handle_reset()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight requests from the Chrome extension."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    # ─── Route handlers ────────────────────────────────────────────

    def _handle_status(self):
        """GET /api/status — Return current server state + system info."""
        with state_lock:
            # Build a serializable copy of state (exclude process object)
            status = {
                "state": server_state["state"],
                "error": server_state["error"],
                "training": server_state["training"],
                "export": server_state["export"],
                "download": server_state["download"],
                "done": server_state["done"],
            }

        # If training is active, refresh from the status file
        if status["state"] == "training":
            live = read_training_status()
            if live:
                status["training"] = live

        # Add system info
        status.update(get_system_info())

        self._send_json(status)

    def _handle_datasets(self):
        """GET /api/datasets — List available JSONL training files."""
        datasets = list_datasets()
        self._send_json({"datasets": datasets})

    def _handle_get_config(self):
        """GET /api/config — Return current config values."""
        config = {
            "model": getattr(cfg, 'MODEL_ID', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
            "max_length": getattr(cfg, 'MAX_LENGTH', 2048),
            "epochs": getattr(cfg, 'NUM_EPOCHS', 3),
            "batch_size": getattr(cfg, 'PER_DEVICE_BATCH_SIZE', 1),
            "grad_accum": getattr(cfg, 'GRAD_ACCUM_STEPS', 16),
            "lr": getattr(cfg, 'LEARNING_RATE', 2e-4),
            "max_hours": getattr(cfg, 'MAX_TRAINING_HOURS', 24),
            "lora_r": getattr(cfg, 'QLORA_R', 64),
            "lora_alpha": getattr(cfg, 'QLORA_ALPHA', 16),
            "lora_dropout": getattr(cfg, 'QLORA_DROPOUT', 0.05),
            "quant_type": getattr(cfg, 'QUANT_TYPE', 'Q8_0'),
        }
        self._send_json(config)

    def _handle_model_check(self):
        """GET /api/model/check?id=... — Check if a model is cached."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        model_id = params.get("id", [""])[0]
        if not model_id:
            self._send_json({"error": "Missing 'id' parameter"}, 400)
            return
        result = check_model_cached(model_id)
        self._send_json(result)

    def _handle_model_download(self):
        """POST /api/model/download — Download a HuggingFace model."""
        try:
            body = self._read_body()
            data = json.loads(body) if body else {}
            model_id = data.get("model_id", "").strip()
            if not model_id:
                self._send_json({"error": "Missing model_id"}, 400)
                return
            if "/" not in model_id:
                self._send_json({"error": f"'{model_id}' doesn't look like a HuggingFace model ID. Expected format: org/model (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)"}, 400)
                return
            start_model_download(model_id)
            self._send_json({"ok": True, "message": f"Downloading {model_id}..."})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_start_training(self):
        """POST /api/training/start — Start a QLoRA training job."""
        try:
            body = self._read_body()
            config = json.loads(body) if body else {}
            start_training(config)
            self._send_json({"ok": True, "message": "Training started"})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_start_export(self):
        """POST /api/export/start — Start a GGUF export job."""
        try:
            body = self._read_body()
            config = json.loads(body) if body else {}
            start_export(config)
            self._send_json({"ok": True, "message": "Export started"})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_stop(self):
        """POST /api/training/stop or /api/export/stop — Stop the current job."""
        stopped = stop_current_job()
        if stopped:
            self._send_json({"ok": True, "message": "Stop signal sent"})
        else:
            self._send_json({"error": "No active job to stop"}, 400)

    def _handle_reset(self):
        """POST /api/reset — Reset server state to idle."""
        with state_lock:
            # Only reset if no process is running
            if server_state.get("process") and server_state["process"].poll() is None:
                self._send_json({"error": "Cannot reset while a job is running"}, 400)
                return

            server_state["state"] = "idle"
            server_state["error"] = None
            server_state["training"] = {}
            server_state["export"] = {}
            server_state["download"] = {}
            server_state["done"] = {}
            server_state["process"] = None
            server_state["process_type"] = None

        self._send_json({"ok": True, "message": "State reset"})

    # ─── Response helpers ──────────────────────────────────────────

    def _send_json(self, data, status=200):
        """Send a JSON response with CORS headers."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _set_cors_headers(self):
        """Allow requests from the Chrome extension."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_body(self):
        """Read the request body (for POST requests)."""
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return self.rfile.read(length).decode("utf-8")
        return ""

    def log_message(self, format, *args):
        """Override to add a [Forge Server] prefix to log output."""
        print(f"[Forge Server] {args[0]} {args[1]} {args[2]}")


# ═══════════════════════════════════════════════════════════════════════
#  SERVER STARTUP
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anthill Forge local API server")
    parser.add_argument("--port", type=int, default=7800,
                        help="Port to listen on (default: 7800)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0, accepts LAN connections)")
    args = parser.parse_args()

    # Make sure output directory exists for the status file
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    server = HTTPServer((args.host, args.port), ForgeHandler)

    print("=" * 55)
    print("  ANTHILL FORGE — Local API Server")
    print("=" * 55)
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Data folder:  {cfg.DATA_FOLDER}")
    print(f"  Output dir:   {cfg.OUTPUT_DIR}")
    if HAS_TORCH and torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    else:
        print(f"  GPU:          Not available")
    print(f"  Press Ctrl+C to stop")
    print("=" * 55)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Forge Server] Shutting down...")
        # Stop any running job gracefully
        stop_current_job()
        server.shutdown()


if __name__ == "__main__":
    main()
