import subprocess
import threading
import time
import requests
import queue
import sys
import json
import re

# ================= CONFIG =================

WHISPER_CMD = [
    "./build/bin/whisper-stream",
    "-m", "./models/ggml-tiny.en.bin",
    "--step", "4000",
    "--length", "8000",
    "-c", "0",
    "-t", "2",
    "-ac", "512",
]

metrics = {
    "speech_end": None,
    "llm_request_start": None,
    "llm_first_token": None,
    "llm_end": None,
    "tokens": 0,
}

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

SILENCE_TIMEOUT = 1.2        # seconds
MIN_CHARS = 25
MAX_PROMPT_CHARS = 512
N_PREDICT = 128

last_prompt_sent = ""


# ==========================================

text_queue = queue.Queue()
stop_event = threading.Event()

listening_enabled = True
llm_busy = False

POST_LLM_COOLDOWN = 1.0  # seconds
last_llm_time = 0

ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def run_whisper():
    global listening_enabled

    proc = subprocess.Popen(
        WHISPER_CMD,
        cwd="whisper.cpp",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    for line in proc.stdout:
        if stop_event.is_set():
            break

        if not listening_enabled:
            continue

        line = line.strip()
        if not line:
            continue

        if "]" in line:
            line = line.split("]", 1)[1].strip()

        text_queue.put(line)

    proc.terminate()


def run_llama_stream(prompt):
    global listening_enabled, llm_busy

    listening_enabled = False
    llm_busy = True

    prompt = prompt[-MAX_PROMPT_CHARS:]
    print("\n? LLM:", end=" ", flush=True)

    try:
        with requests.post(
            LLAMA_URL,
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a voice assistant running on a Raspberry Pi.\n\n"
                            "Rules:\n"
                            "- Respond in 1 short sentence ONLY.\n"
                            "- Do NOT mention model names, training data, or developers.\n"
                            "- If the answer would be long, summarize aggressively.\n"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "n_predict": N_PREDICT,
                "temperature": 0.5,
                "stream": True
            },
            stream=True,
            timeout=None
        ) as r:
            r.raise_for_status()

            for raw_line in r.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                if payload == "[DONE]":
                    break

                event = json.loads(payload)
                delta = event["choices"][0]["delta"].get("content", "")
                if delta:
                    if metrics["llm_first_token"] is None:
                        metrics["llm_first_token"] = time.monotonic()
                    metrics["tokens"] += 1
                    print(delta, end="", flush=True)

        print("\n")

    except Exception as e:
        print(f"\n[LLAMA ERROR] {e}\n")

    finally:
        llm_busy = False
        listening_enabled = True
        metrics["llm_end"] = time.monotonic()
        print_metrics()
        global last_llm_time
        last_llm_time = time.monotonic()




def text_aggregator():
    global llm_busy, last_prompt_sent

    buffer = ""
    last_update = time.monotonic()

    while not stop_event.is_set():
        try:
            text = text_queue.get(timeout=0.1)

            # Ignore ASR while LLM is active
            if llm_busy:
                continue

            text = text.strip()
            if not text:
                continue

            buffer += " " + text
            last_update = time.monotonic()
            print(f"\r?? {buffer}", end="", flush=True)

        except queue.Empty:
            pass

        # Do nothing while LLM is busy
        if llm_busy:
            continue

        # Silence-based finalization
        if buffer and (time.monotonic() - last_update) > SILENCE_TIMEOUT:
            # Cooldown after LLM response (prevents feedback loops)
            if time.monotonic() - last_llm_time < POST_LLM_COOLDOWN:
                buffer = ""
                last_update = time.monotonic()
                continue

            clean = buffer.strip()

            # Length guard
            if len(clean) < MIN_CHARS:
                buffer = ""
                last_update = time.monotonic()
                continue

            # Duplicate prompt guard
            if clean == last_prompt_sent:
                buffer = ""
                last_update = time.monotonic()
                continue

            # Commit prompt
            last_prompt_sent = clean
            print("\n\n? Sending to LLM...")
            metrics["llm_request_start"] = time.monotonic()

            run_llama_stream(clean)
            print("\n---\n")

            # HARD RESET after LLM turn
            buffer = ""
            last_update = time.monotonic()

            # Flush any stale ASR output

    


def print_metrics():
    if not all([
        metrics["speech_end"],
        metrics["llm_request_start"],
        metrics["llm_first_token"],
        metrics["llm_end"]
    ]):
        return

    asr_to_llm = metrics["llm_request_start"] - metrics["speech_end"]
    ttft = metrics["llm_first_token"] - metrics["llm_request_start"]
    llm_duration = metrics["llm_end"] - metrics["llm_first_token"]
    tps = metrics["tokens"] / llm_duration if llm_duration > 0 else 0
    end_to_end = metrics["llm_first_token"] - metrics["speech_end"]

    print("\nTiming Metrics")
    print(f"ASR → LLM decision latency : {asr_to_llm:.3f}s")
    print(f"LLM Time-to-First-Token   : {ttft:.3f}s")
    print(f"LLM Tokens/sec            : {tps:.2f}")
    print(f"End-to-End Latency        : {end_to_end:.3f}s")


def main():
    whisper_thread = threading.Thread(target=run_whisper, daemon=True)
    agg_thread = threading.Thread(target=text_aggregator, daemon=True)


    whisper_thread.start()
    agg_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        sys.exit(0)


if __name__ == "__main__":
    main()
