# loom_server.py — Local HTTP API server for Anthill Loom
#
# This lightweight server acts as a bridge between the Anthill Loom Chrome
# extension (the dashboard) and the local data processing scripts.
# It runs on localhost:7801 and provides a REST API for:
#   - Listing raw Spider exports and processed output files
#   - Starting the extraction + dedup pipeline
#   - Reporting processing progress and results
#
# Usage:
#   python loom_server.py              # start on default port 7801
#   python loom_server.py --port 8888  # custom port
#
# No external dependencies — uses Python stdlib only.

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Where Spider dumps raw conversation exports
RAW_DIR = SCRIPT_DIR / "datasets" / "raw"

# Where Loom writes processed JSONL files (also where Forge reads from)
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

# The two processing scripts
EXTRACT_SCRIPT = SCRIPT_DIR / "prepare_datasets_parallel.py"
DEDUPE_SCRIPT = SCRIPT_DIR / "dedupe_and_filter.py"

# Also check Downloads folder for Spider exports dropped there
DOWNLOADS_DIR = Path.home() / "Downloads"


# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL STATE — tracks what the server is currently doing
# ═══════════════════════════════════════════════════════════════════════

server_state = {
    "state": "idle",       # idle, processing, done, error
    "error": None,
    "processing": {},      # progress info during processing
    "done": {},            # results after processing completes
}
state_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
#  FILE DISCOVERY — find raw inputs and processed outputs
# ═══════════════════════════════════════════════════════════════════════

def find_raw_files():
    """
    Find all raw ChatGPT conversation JSON files.
    Looks in both the datasets/raw directory and the Downloads folder.
    """
    files = []

    # Check datasets/raw/
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.glob("*.json")):
            files.append(_file_info(f))

    # Check Downloads folder for chatgpt_conversations_*.json
    if DOWNLOADS_DIR.exists():
        for f in sorted(DOWNLOADS_DIR.glob("chatgpt_conversations_*.json")):
            # Don't double-list if it's already in raw/
            if not any(existing["path"] == str(f) for existing in files):
                files.append(_file_info(f))

    return files


# Cache for expensive file metadata (conversation counts, line counts).
# Keyed by (filepath, mtime) so it auto-invalidates when files change.
_file_cache = {}


def _file_info(path):
    """Build a metadata dict for a file. Uses cache to avoid re-reading large files."""
    stat = path.stat()
    size_mb = round(stat.st_size / (1024 * 1024), 1)
    cache_key = (str(path), stat.st_mtime)

    # Return cached result if the file hasn't changed
    if cache_key in _file_cache:
        return _file_cache[cache_key]

    info = {
        "name": path.name,
        "path": str(path),
        "size_mb": size_mb,
        "conversations": None,
    }

    # Estimate conversation count by counting top-level array items
    # without loading the entire file into memory. We just count
    # lines that start with a pattern indicating a new conversation object.
    # For huge files (>10MB), skip counting entirely — it's not worth the wait.
    if stat.st_size < 10 * 1024 * 1024:  # only for files under 10MB
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                convos = data.get("conversations", [])
                info["conversations"] = len(convos)
        except Exception:
            pass
    else:
        # For large files, give a rough estimate from file size
        # (average ~120KB per conversation based on typical ChatGPT exports)
        info["conversations"] = round(stat.st_size / (120 * 1024))

    _file_cache[cache_key] = info
    return info


def find_output_files():
    """Find all processed JSONL files in the output directory."""
    files = []

    if not PROCESSED_DIR.exists():
        return files

    for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
        stat = f.stat()
        cache_key = (str(f), stat.st_mtime)

        if cache_key in _file_cache:
            files.append(_file_cache[cache_key])
            continue

        info = {
            "name": f.name,
            "path": str(f),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
        }

        # Count lines (= pairs) in the JSONL file
        try:
            with open(f, "r", encoding="utf-8") as fh:
                info["pairs"] = sum(1 for _ in fh)
        except Exception:
            info["pairs"] = None

        _file_cache[cache_key] = info
        files.append(info)

    return files


def get_total_conversations():
    """Count total conversations across all raw files."""
    total = 0
    raw = find_raw_files()
    for f in raw:
        if f.get("conversations"):
            total += f["conversations"]
    return total


def _unique_path(path):
    """If path already exists, append a number to avoid overwriting.
    e.g. pairs_clean.jsonl -> pairs_clean_2.jsonl -> pairs_clean_3.jsonl"""
    if not path.exists():
        return path
    stem = path.stem     # "pairs_clean"
    suffix = path.suffix # ".jsonl"
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


# ═══════════════════════════════════════════════════════════════════════
#  PROCESSING — run the extraction + dedup pipeline
# ═══════════════════════════════════════════════════════════════════════

def start_processing(filename=None, use_latest=False):
    """
    Run the Loom pipeline: extract pairs then deduplicate.
    If filename is given, process that specific file.
    If use_latest is True, find the most recent Spider export.
    """
    with state_lock:
        if server_state["state"] == "processing":
            raise RuntimeError("Already processing — wait for current job to finish")

    # Find the input file
    if use_latest:
        raw = find_raw_files()
        if not raw:
            raise FileNotFoundError("No raw JSON files found in datasets/raw/ or Downloads")
        # Pick the largest (most likely the latest full export)
        input_file = max(raw, key=lambda f: f["size_mb"])
        input_path = Path(input_file["path"])
    elif filename:
        # Find the file by name across known locations
        raw = find_raw_files()
        match = [f for f in raw if f["name"] == filename]
        if not match:
            raise FileNotFoundError(f"File not found: {filename}")
        input_path = Path(match[0]["path"])
    else:
        raise ValueError("Specify a filename or use_latest=True")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set up output paths — avoid overwriting existing files
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = _unique_path(PROCESSED_DIR / "pairs.jsonl")
    clean_output = _unique_path(PROCESSED_DIR / "pairs_clean.jsonl")

    with state_lock:
        server_state["state"] = "processing"
        server_state["error"] = None
        server_state["processing"] = {
            "phase": "Starting...",
            "message": f"Processing {input_path.name}",
            "detail": "Loading file...",
            "progress": 5,
        }

    # Run pipeline in a background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(input_path, raw_output, clean_output),
        daemon=True,
    )
    thread.start()

    return True


def _run_pipeline(input_path, raw_output, clean_output):
    """
    Background thread that runs both processing steps:
    1. prepare_datasets_parallel.py — extract user/assistant pairs
    2. dedupe_and_filter.py — remove duplicates and junk
    """
    try:
        # ── Step 1: Extract pairs ──────────────────────────────────
        with state_lock:
            server_state["processing"] = {
                "phase": "Step 1/2: Extracting pairs",
                "message": f"Processing {input_path.name}...",
                "detail": "Extracting user/assistant conversation pairs",
                "progress": 20,
            }

        print(f"[Loom Server] Step 1: Extracting pairs from {input_path.name}")

        result1 = subprocess.run(
            [sys.executable, str(EXTRACT_SCRIPT), str(input_path), str(raw_output)],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR),
            timeout=300,  # 5 min timeout (should be seconds for most files)
        )

        if result1.returncode != 0:
            error_msg = result1.stderr.strip() or result1.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Extraction failed: {error_msg}")

        # Parse extraction output for stats
        extract_output = result1.stdout
        raw_pairs = _count_lines(raw_output)
        conversations = _parse_stat(extract_output, "Found", "conversations")
        total_messages = _parse_stat(extract_output, "Total messages:", None)

        print(f"[Loom Server] Step 1 complete: {raw_pairs} raw pairs extracted")

        with state_lock:
            server_state["processing"] = {
                "phase": "Step 2/2: Deduplicating",
                "message": f"Extracted {raw_pairs:,} raw pairs, now cleaning...",
                "detail": "Removing duplicates and filtering short/long pairs",
                "progress": 60,
            }

        # ── Step 2: Deduplicate and filter ─────────────────────────
        print(f"[Loom Server] Step 2: Deduplicating and filtering")

        result2 = subprocess.run(
            [sys.executable, str(DEDUPE_SCRIPT), str(raw_output), str(clean_output)],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR),
            timeout=300,
        )

        if result2.returncode != 0:
            error_msg = result2.stderr.strip() or result2.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Deduplication failed: {error_msg}")

        clean_pairs = _count_lines(clean_output)
        dupes_removed = raw_pairs - clean_pairs

        print(f"[Loom Server] Step 2 complete: {clean_pairs} clean pairs ({dupes_removed} removed)")

        # ── Done ───────────────────────────────────────────────────
        with state_lock:
            server_state["state"] = "done"
            server_state["processing"] = {}
            server_state["done"] = {
                "message": f"Extracted <strong>{clean_pairs:,}</strong> clean training pairs from {input_path.name}",
                "input_file": input_path.name,
                "output_file": clean_output.name,
                "conversations": conversations,
                "total_messages": total_messages,
                "raw_pairs": raw_pairs,
                "clean_pairs": clean_pairs,
                "duplicates_removed": dupes_removed,
            }

        print(f"[Loom Server] Pipeline complete: {clean_pairs} clean pairs ready for Forge")

    except Exception as e:
        print(f"[Loom Server] Pipeline error: {e}")
        with state_lock:
            server_state["state"] = "error"
            server_state["error"] = str(e)
            server_state["processing"] = {}


def _count_lines(path):
    """Count lines in a file (= number of JSONL entries)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _parse_stat(output, prefix, suffix):
    """Try to extract a number from script output like 'Found 531 conversations'."""
    try:
        for line in output.split("\n"):
            if prefix in line:
                # Extract the first number after the prefix
                parts = line.split(prefix)[-1].strip().split()
                for part in parts:
                    cleaned = part.replace(",", "").strip()
                    if cleaned.isdigit():
                        return int(cleaned)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
#  HTTP REQUEST HANDLER
# ═══════════════════════════════════════════════════════════════════════

class LoomHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests from the Loom Chrome extension."""

    def handle(self):
        """Override handle to suppress ConnectionAbortedError on Windows.
        This happens when the extension's fetch times out before we finish
        sending — harmless, but spams the console without this catch."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        if self.path == '/api/status':
            self._handle_status()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == '/api/process':
            self._handle_process()
        elif self.path == '/api/reset':
            self._handle_reset()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    # ─── Route handlers ────────────────────────────────────────────

    def _handle_status(self):
        """GET /api/status — return current state, file lists, stats."""
        with state_lock:
            status = {
                "state": server_state["state"],
                "error": server_state["error"],
                "processing": server_state["processing"],
                "done": server_state["done"],
            }

        # Add file listings (always useful for the dashboard)
        status["raw_files"] = find_raw_files()
        status["output_files"] = find_output_files()
        status["total_conversations"] = get_total_conversations()

        self._send_json(status)

    def _handle_process(self):
        """POST /api/process — start the extraction pipeline."""
        try:
            body = self._read_body()
            config = json.loads(body) if body else {}

            filename = config.get("filename")
            use_latest = config.get("latest", False)

            start_processing(filename=filename, use_latest=use_latest)
            self._send_json({"ok": True, "message": "Processing started"})

        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_reset(self):
        """POST /api/reset — reset state to idle."""
        with state_lock:
            server_state["state"] = "idle"
            server_state["error"] = None
            server_state["processing"] = {}
            server_state["done"] = {}
        self._send_json({"ok": True})

    # ─── Response helpers ──────────────────────────────────────────

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return self.rfile.read(length).decode("utf-8")
        return ""

    def log_message(self, format, *args):
        print(f"[Loom Server] {args[0]} {args[1]} {args[2]}")


# ═══════════════════════════════════════════════════════════════════════
#  SERVER STARTUP
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anthill Loom local API server")
    parser.add_argument("--port", type=int, default=7801,
                        help="Port to listen on (default: 7801)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    # Make sure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Count what we have to start with
    raw = find_raw_files()
    output = find_output_files()

    server = HTTPServer((args.host, args.port), LoomHandler)

    print("=" * 55)
    print("  ANTHILL LOOM — Local API Server")
    print("=" * 55)
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Raw files:    {RAW_DIR} ({len(raw)} files)")
    print(f"  Output dir:   {PROCESSED_DIR} ({len(output)} files)")
    print(f"  Downloads:    {DOWNLOADS_DIR}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 55)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Loom Server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
