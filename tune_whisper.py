#!/usr/bin/env python3
"""
whisper.cpp stream parameter sweep for Raspberry Pi 4
"""

import subprocess
import time
import itertools
import json
import re
import sys
from datetime import datetime

# ── Paths 
WHISPER_BIN = "./build/bin/whisper-stream"
MODEL       = "./models/ggml-tiny.en.bin"
WHISPER_DIR = "../whisper.cpp"

# ── How long to observe each config (seconds) ─────────────────────────────────
TEST_DURATION = 20

# ── Parameter grid ─────────────────────────────────────────────────────────────
PARAM_GRID = {
    "step":    [500, 1000, 2000],   # ms — added 500/1000 for snappier feel
    "length":  [3000, 5000, 7000],  # ms — shorter windows = less RAM pressure
    "threads": [2, 3],
    "ac":      [256, 512, 768],     # --audio-ctx
}

# ── Results log ────────────────────────────────────────────────────────────────
RESULTS_FILE = "sweep_results.json"
results: list[dict] = []


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_cpu_times() -> tuple[float, float]:
    """Return (idle, total) jiffies from /proc/stat for CPU0+CPU1."""
    with open("/proc/stat") as f:
        lines = f.readlines()
    idle = total = 0.0
    for line in lines[1:3]:          # cpu0, cpu1 — the cores we pin to
        parts = line.split()
        vals  = [float(x) for x in parts[1:]]
        idle  += vals[3]             # idle field
        total += sum(vals)
    return idle, total


def cpu_percent(idle0: float, total0: float,
                idle1: float, total1: float) -> float:
    d_idle  = idle1  - idle0
    d_total = total1 - total0
    if d_total == 0:
        return 0.0
    return round(100.0 * (1.0 - d_idle / d_total), 1)


def parse_latency(line: str) -> float | None:
    """
    whisper-stream prints timing lines like:
      whisper_print_timings:   total time =  1234.56 ms
    Return the ms value if found, else None.
    """
    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", line)
    return float(m.group(1)) if m else None


def run_whisper(params: dict) -> dict:
    """Launch whisper-stream, observe for TEST_DURATION s, return metrics."""
    cmd = [
        "taskset", "-c", "0,1",
        WHISPER_BIN,
        "-m",       MODEL,
        "--step",   str(params["step"]),
        "--length", str(params["length"]),
        "-t",       str(params["threads"]),
        "-ac",      str(params["ac"]),  
        "--no-timestamps",               # saves CPU — no timestamp math
        "-vth",     "0.6",              # VAD: skip silence chunks
        "-c",       "0",                # capture device
    ]

    print("\n" + "═" * 50)
    print(f"  CONFIG: step={params['step']}ms  length={params['length']}ms  "
          f"threads={params['threads']}  audio-ctx={params['ac']}")
    print("═" * 50)

    idle0, total0 = read_cpu_times()
    proc = subprocess.Popen(
        cmd,
        cwd=WHISPER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,       # capture stderr — whisper timings go here
        text=True,
        bufsize=1,
    )

    latencies:    list[float] = []
    output_lines: list[str]   = []
    start = time.time()

    try:
        # Read stdout (transcription) and stderr (timings) concurrently via
        # a simple non-blocking poll on stderr between stdout reads.
        import select, os

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
                    # Strip timestamp brackets from transcript lines
                    if "]" in line and fd == proc.stdout.fileno():
                        line = line.split("]", 1)[1].strip()
                    if line:
                        print(f"  {line}")
                        output_lines.append(line)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    idle1, total1 = read_cpu_times()
    cpu = cpu_percent(idle0, total0, idle1, total1)

    avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else None
    min_lat = round(min(latencies), 1)                   if latencies else None

    metrics = {
        "params":        params,
        "cpu_pct_cores01": cpu,
        "avg_latency_ms":  avg_lat,
        "min_latency_ms":  min_lat,
        "inference_count": len(latencies),
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }

    print(f"\n  📊  CPU (cores 0-1): {cpu}%  |  "
          f"Avg latency: {avg_lat} ms  |  "
          f"Inferences: {len(latencies)}")
    return metrics


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(m: dict) -> float:
    """
    Lower is better.
    Penalise high CPU and high latency equally.
    Returns a simple combined score (both normalised to ~0-100 range).
    """
    cpu = m.get("cpu_pct_cores01") or 100.0
    lat = m.get("avg_latency_ms")  or 9999.0
    # weight CPU heavier since you have two other processes competing
    return cpu * 1.5 + lat / 50.0


def print_leaderboard(results: list[dict]) -> None:
    ranked = sorted(results, key=score)
    print("\n" + "═" * 60)
    print("  LEADERBOARD  (lower score = better for Pi 4 workload)")
    print("═" * 60)
    print(f"  {'Rank':<5} {'Step':>5} {'Len':>6} {'Thr':>4} {'AC':>5} "
          f"{'CPU%':>6} {'AvgLat':>8} {'Score':>7}")
    print("  " + "─" * 56)
    for i, m in enumerate(ranked[:10], 1):
        p = m["params"]
        print(f"  {i:<5} {p['step']:>5} {p['length']:>6} {p['threads']:>4} "
              f"{p['ac']:>5} {m['cpu_pct_cores01']:>6} "
              f"{str(m['avg_latency_ms'])+' ms':>8} "
              f"{score(m):>7.1f}")
    print()
    best = ranked[0]
    bp   = best["params"]
    print("  ✅  RECOMMENDED COMMAND:")
    print(f"""
  taskset -c 0,1 {WHISPER_BIN} \\
    -m {MODEL} \\
    --step {bp['step']} \\
    --length {bp['length']} \\
    -t {bp['threads']} \\
    -ac {bp['ac']} \\
    --no-timestamps \\
    -vth 0.6 \\
    -c 0
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def parameter_sweep(interactive: bool = True) -> None:
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)

    print(f"\n  whisper.cpp stream sweep — {total} configurations")
    print(f"  Each test runs for {TEST_DURATION}s. Speak into the mic!")
    print(f"  Results saved to: {RESULTS_FILE}\n")

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        print(f"\n  [{idx}/{total}]", end="")

        metrics = run_whisper(params)
        results.append(metrics)

        # Persist after every run so you can Ctrl-C and keep progress
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        if interactive and idx < total:
            print("\n  Press ENTER for next test  (or 's' to skip to summary)…")
            resp = input("  > ").strip().lower()
            if resp == "s":
                break

    print_leaderboard(results)


if __name__ == "__main__":
    interactive = "--auto" not in sys.argv   # pass --auto to skip pauses
    parameter_sweep(interactive=interactive)
