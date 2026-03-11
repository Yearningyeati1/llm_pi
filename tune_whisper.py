#!/usr/bin/env python3
"""
whisper.cpp stream parameter sweep — Raspberry Pi 4 edition
Flags validated against whisper-stream --help output.
"""

import subprocess
import time
import itertools
import json
import re
import sys
import os
import select
from datetime import datetime

# ── Paths (from your original script) ─────────────────────────────────────────
WHISPER_BIN = "./build/bin/whisper-stream"
MODEL       = "./models/ggml-tiny.en.bin"
WHISPER_DIR = "../whisper.cpp"

# ── Observation window per config ─────────────────────────────────────────────
TEST_DURATION = 20      # seconds — speak into the mic during this window

# ── Parameter grid ─────────────────────────────────────────────────────────────
PARAM_GRID = {
    "step":       [500, 1000, 2000],    # --step        chunk cadence (ms)
    "length":     [3000, 5000, 7000],   # --length      context window (ms)
    "keep":       [100, 200],           # --keep        overlap between chunks (ms)
    "ac":         [256, 512],           # --audio-ctx   encoder context
    "max_tokens": [16, 32],             # --max-tokens  limits decoder iterations
    "beam_size":  [-1, 1],             # --beam-size   -1=greedy(fast), 1=beam
    "threads":    [2, 3],              # --threads
}

# ── Fixed flags — never sweep these on Pi 4 ───────────────────────────────────
FIXED_FLAGS = [
    "--language",   "en",       # skip language-detection overhead
    "--vad-thold",  "0.6",      # skip silent chunks
    "--freq-thold", "100.0",    # high-pass filter, reduces noise load
    "--no-fallback",            # skip temperature fallback retries
    "--no-gpu",                 # Pi 4 has no usable GPU, be explicit
    # --flash-attn is ON by default; leave it, it helps on ARM NEON
]

RESULTS_FILE = "sweep_results.json"
results = []


# ── CPU measurement (cores 0-1 only, matching taskset) ────────────────────────

def read_cpu_times():
    with open("/proc/stat") as f:
        lines = f.readlines()
    idle = total = 0.0
    for line in lines[1:3]:        # cpu0, cpu1
        parts = line.split()
        vals  = [float(x) for x in parts[1:]]
        idle  += vals[3]
        total += sum(vals)
    return idle, total


def cpu_percent(s0, s1):
    d_idle  = s1[0] - s0[0]
    d_total = s1[1] - s0[1]
    if d_total == 0:
        return 0.0
    return round(100.0 * (1.0 - d_idle / d_total), 1)


# ── Latency parser ─────────────────────────────────────────────────────────────

def parse_latency(line):
    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", line)
    return float(m.group(1)) if m else None


# ── Single test run ────────────────────────────────────────────────────────────

def run_whisper(params):
    cmd = [
        "taskset", "-c", "0,1",
        WHISPER_BIN,
        "-m",            MODEL,
        "--step",        str(params["step"]),
        "--length",      str(params["length"]),
        "--keep",        str(params["keep"]),
        "--threads",     str(params["threads"]),
        "--audio-ctx",   str(params["ac"]),
        "--max-tokens",  str(params["max_tokens"]),
        "--beam-size",   str(params["beam_size"]),
        "--capture",     "0",
    ] + FIXED_FLAGS

    print("\n" + "=" * 58)
    print(f"  step={params['step']}ms  length={params['length']}ms  keep={params['keep']}ms  threads={params['threads']}")
    print(f"  audio-ctx={params['ac']}  max-tokens={params['max_tokens']}  beam-size={params['beam_size']}")
    print("=" * 58)

    cpu_before = read_cpu_times()
    proc = subprocess.Popen(
        cmd,
        cwd=WHISPER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    latencies = []
    start = time.time()

    try:
        fds = [proc.stdout.fileno(), proc.stderr.fileno()]
        while time.time() - start < TEST_DURATION:
            ready, _, _ = select.select(fds, [], [], 0.5)
            for fd in ready:
                raw = os.read(fd, 4096).decode("utf-8", errors="replace")
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    lat = parse_latency(line)
                    if lat:
                        latencies.append(lat)
                    if fd == proc.stdout.fileno():
                        display = line.split("]", 1)[-1].strip() if "]" in line else line
                        if display:
                            print(f"  > {display}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    cpu_after = read_cpu_times()
    cpu  = cpu_percent(cpu_before, cpu_after)
    avg  = round(sum(latencies) / len(latencies), 1) if latencies else None
    best = round(min(latencies), 1)                   if latencies else None

    print(f"\n  CPU (cores 0-1): {cpu}%  |  Avg latency: {avg} ms  |  Best: {best} ms  |  Inferences: {len(latencies)}")

    return {
        "params":           params,
        "cpu_pct":          cpu,
        "avg_latency_ms":   avg,
        "best_latency_ms":  best,
        "inference_count":  len(latencies),
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
    }


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(m):
    cpu = m.get("cpu_pct")        or 100.0
    lat = m.get("avg_latency_ms") or 9999.0
    # CPU weighted 1.5x — llama.cpp + piper share the same 4 cores
    return cpu * 1.5 + lat / 50.0


# ── Leaderboard ────────────────────────────────────────────────────────────────

def print_leaderboard(results):
    ranked = sorted(results, key=score)
    print("\n\n" + "=" * 72)
    print("  LEADERBOARD  —  lower score = lighter on Pi 4 resources")
    print("=" * 72)
    print(f"  {'#':<3} {'step':>5} {'len':>6} {'keep':>5} {'thr':>4} {'ac':>5} {'mt':>4} {'bs':>4} {'CPU%':>6} {'AvgLat':>8} {'Score':>7}")
    print("  " + "-" * 68)

    for i, m in enumerate(ranked[:10], 1):
        p = m["params"]
        lat_str = f"{m['avg_latency_ms']}ms" if m["avg_latency_ms"] else "  n/a"
        print(f"  {i:<3} {p['step']:>5} {p['length']:>6} {p['keep']:>5} "
              f"{p['threads']:>4} {p['ac']:>5} {p['max_tokens']:>4} "
              f"{p['beam_size']:>4} {m['cpu_pct']:>6} {lat_str:>8} "
              f"{score(m):>7.1f}")

    best = ranked[0]
    bp   = best["params"]
    print(f"""
  RECOMMENDED COMMAND:

  taskset -c 0,1 {WHISPER_BIN} \\
    -m {MODEL} \\
    --step        {bp['step']} \\
    --length      {bp['length']} \\
    --keep        {bp['keep']} \\
    --threads     {bp['threads']} \\
    --audio-ctx   {bp['ac']} \\
    --max-tokens  {bp['max_tokens']} \\
    --beam-size   {bp['beam_size']} \\
    --language    en \\
    --vad-thold   0.6 \\
    --freq-thold  100.0 \\
    --no-fallback \\
    --no-gpu \\
    --capture     0
""")


# ── Sweep ──────────────────────────────────────────────────────────────────────

def parameter_sweep(interactive=True):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)

    print(f"\n  whisper-stream parameter sweep  —  {total} configurations")
    print(f"  {TEST_DURATION}s per test  |  speak into the mic each round")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"  Run with --auto to skip prompts\n")

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        print(f"\n  -- Test {idx} / {total} --", end="")

        metrics = run_whisper(params)
        results.append(metrics)

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        if interactive and idx < total:
            print("\n  ENTER = next   s = skip to summary   q = quit")
            resp = input("  > ").strip().lower()
            if resp in ("q", "s"):
                break

    print_leaderboard(results)


if __name__ == "__main__":
    interactive = "--auto" not in sys.argv
    parameter_sweep(interactive=interactive)
