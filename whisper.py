import subprocess
import threading
import time
import requests
import queue
import sys
import json
import re
# ================= Add Piper TTS ============= 06/03/2026
PIPER_MODEL = "./models/en_US-lessac-low.onnx"
PIPER_BIN   = "./piper/piper"
CHUNK_PATTERN = re.compile(r'([^.!?,]+[.!?,]+)')
tts_queue = queue.Queue()

def piper_speak(text):
    """Speak a single chunk via piper."""
    try:
        proc = subprocess.Popen(
            [
                PIPER_BIN,
                "--model", PIPER_MODEL,
                "--output-raw",          # raw PCM to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # pipe text in, get raw PCM out → play with aplay
        aplay = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=proc.stdout,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode())
        proc.stdin.close()
        proc.wait()
        aplay.wait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")


def tts_worker():
    """Dedicated thread that speaks chunks sequentially."""
    while True:
        chunk = tts_queue.get()
        if chunk is None:   # poison pill — stop signal
            break
        piper_speak(chunk)
        tts_queue.task_done()
# +++++++++++++++++++++++++++++++++++++++++++++++++
# ================= CONFIG =================

WHISPER_CMD = [
    "taskset", "-c", "0,1",
    "./build/bin/whisper-stream",
    "-m", "./models/ggml-tiny.en.bin",
    "--step", "3000", # 4000
    "--length", "6000", # 8000
    "-c", "0",
    "-t", "2", # 4
    "-ac", "512", # 512
]

metrics = {
    "speech_end": None,
    "llm_request_start": None,
    "llm_first_token": None,
    "llm_end": None,
    "tts_end": None,
    "tokens": 0,
}

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

SILENCE_TIMEOUT = 1.2 #1.2        # seconds
MIN_CHARS = 25
MAX_PROMPT_CHARS = 512
N_PREDICT = 40

last_prompt_sent = ""


# ==========================================

text_queue = queue.Queue()
stop_event = threading.Event()

listening_enabled = True
llm_busy = False

POST_LLM_COOLDOWN = 1.0  # seconds
last_llm_time = 0

ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def speak(text):
    """
    Blocking TTS using espeak-ng
    """
    try:
        subprocess.run(
            ["espeak-ng", "-s", "165", "-v", "en-us", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception as e:
        print(f"[TTS ERROR] {e}")

# Add global TCP session 06/03/2026

session = requests.Session()
# == System Prompt ++ 06/03
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a voice assistant named CARAT developed at NIT Rourkela. Reply briefly."
}
#===========================
# -------- LLM Warmup exercise --------------- 06/03.2026

def warmup_llm():
    try:
        session.post(
            LLAMA_URL,
            json={
                "messages": [
                    SYSTEM_MESSAGE,
                    {"role": "user", "content": "hi"}
                ],
                "n_predict": 1,
                "stream": False,
                "cache_prompt": True
            },
            timeout=10
        )
        print("[WARMUP] LLM ready")
    except Exception as e:
        print(f"[WARMUP ERROR] {e}")
# --------  ---------------

def run_whisper():
    global listening_enabled

    print("Listening: ")

    proc = subprocess.Popen(
        WHISPER_CMD,
        cwd = "../whisper.cpp",
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


def run_llama_stream(prompt,speech_end_time):
    global listening_enabled, llm_busy
    # ------ Metrics bug 05/03/2026 ---------------
    metrics["speech_end"] = speech_end_time
    metrics["llm_request_start"] = time.monotonic()
    metrics["llm_first_token"] = None
    metrics["llm_end"] = None
    metrics["tokens"] = 0
    # --------------------------------------------------
    listening_enabled = False
    llm_busy = True

    prompt = prompt[-MAX_PROMPT_CHARS:]
    spoken_text = ""

    # ======= Additions for Piper TTS === 06/03/2026
    partial     = ""          # accumulates tokens until chunk boundary

    # start TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    # ====================================

    try:
        with session.post(
            LLAMA_URL,
            json={
                "messages": [
                    SYSTEM_MESSAGE,
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "n_predict": N_PREDICT,
                "temperature": 0.7,
                "cache_prompt": True,
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
                    spoken_text += delta
                    #print(delta, end="", flush=True)
# -------------- Piper -----------------------------
                    partial     += delta
                    print(delta, end="", flush=True)
                    matches = CHUNK_PATTERN.findall(partial)
                    for match in matches:
                        chunk = match.strip()
                        if chunk:
                            tts_queue.put(chunk)
                    # keep unmatched remainder in partial
                    partial = CHUNK_PATTERN.sub("", partial)
        if partial.strip():
            tts_queue.put(partial.strip())
#------------------------------------------------------


        print("\n")

    except Exception as e:
        print(f"\n[LLAMA ERROR] {e}\n")

    finally:
        # Speak response (TTS)
        # if spoken_text.strip():
        #     speak(spoken_text.strip())
        #     metrics["tts_end"] = time.monotonic()
        # llm_busy = False
        # listening_enabled = True
        # print("Listening: ")
        # metrics["llm_end"] = time.monotonic()
        # print_metrics()
        # global last_llm_time
        # last_llm_time = time.monotonic()
        # --- Piper is the new boss in town

        tts_queue.put(None)      # signal worker to stop
        tts_thread.join()        # wait for all chunks to finish speaking
        metrics["tts_end"] = time.monotonic()

        llm_busy = False
        listening_enabled = True
        print("Listening: ")
        metrics["llm_end"] = time.monotonic()
        print_metrics()

        global last_llm_time
        last_llm_time = time.monotonic()
# ------------------------------------------------------------



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
            # last_prompt_sent = clean
            # metrics["speech_end"] = time.monotonic()
            # print("\n\n? Sending to LLM...")
            # metrics["llm_request_start"] = time.monotonic()

            # ------ 05/03/2026 metrics have messed stuff up
            speech_end = time.monotonic()         # capture locally
            print("\n\n? Sending to LLM...")
            run_llama_stream(clean, speech_end)   # pass it in

            # -------

            # run_llama_stream(clean)
            # print("\n---\n")

            # HARD RESET after LLM turn
            buffer = ""
            last_update = time.monotonic()

            # Flush any stale ASR output

    


# def print_metrics():
#     if not all([
#         metrics["speech_end"],
#         metrics["llm_request_start"],
#         metrics["llm_first_token"],
#         metrics["llm_end"]
#     ]):
#         return

#     asr_to_llm = metrics["llm_request_start"] - metrics["speech_end"]
#     ttft = metrics["llm_first_token"] - metrics["llm_request_start"]
#     llm_duration = metrics["llm_end"] - metrics["llm_first_token"]
#     tps = metrics["tokens"] / llm_duration if llm_duration > 0 else 0
#     end_to_end = metrics["llm_first_token"] - metrics["speech_end"]

#     print("\nTiming Metrics")
#     print(f"ASR → LLM decision latency : {asr_to_llm:.3f}s")
#     print(f"LLM Time-to-First-Token   : {ttft:.3f}s")
#     print(f"LLM Tokens/sec            : {tps:.2f}")
#     print(f"End-to-End Latency        : {end_to_end:.3f}s")

# --------- Update print func
def print_metrics():
    try:
        asr_to_llm = metrics["llm_request_start"] - metrics["speech_end"]
        ttft = metrics["llm_first_token"] - metrics["llm_request_start"]
        llm_duration = metrics["llm_end"] - metrics["llm_first_token"]
        tps = metrics["tokens"] / llm_duration if llm_duration > 0 else 0
        end_to_end    = metrics["tts_end"]           - metrics["speech_end"]  # ← full pipeline

        print("\nTiming Metrics")
        print(f"ASR → LLM decision latency : {asr_to_llm:.3f}s")
        print(f"LLM Time-to-First-Token    : {ttft:.3f}s")
        print(f"LLM Tokens/sec             : {tps:.2f}")
        print(f"End-to-End Latency         : {end_to_end:.3f}s")
    except TypeError as e:
        print(f"[METRICS ERROR] Missing value: {e}")


def main():
    warmup_llm()
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
