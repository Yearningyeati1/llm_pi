"""
CARAT Voice Assistant
Developed at NIT Rourkela

Pipeline: Whisper ASR → LLaMA LLM (streaming) → Piper TTS
"""

import json
import logging
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("carat")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # Piper TTS
    piper_bin: str   = "/home/pi/Desktop/code_llm_pi/piper/piper"
    piper_model: str = "/home/pi/Desktop/code_llm_pi/en_US-ryan-low.onnx"

    # Whisper ASR
    whisper_bin: str   = "./build/bin/whisper-stream"
    whisper_model: str = "./models/ggml-tiny.en.bin"
    whisper_cwd: str   = "../whisper.cpp"
    whisper_step: int  = 2000
    whisper_length: int = 5000
    whisper_threads: int = 2
    whisper_audio_ctx: int = 400
    whisper_card: int = 0
    whisper_cpu_cores: str = "0,1"

    # LLaMA
    llama_url: str    = "http://localhost:8080/v1/chat/completions"
    n_predict: int    = 40
    temperature: float = 0.4
    max_prompt_chars: int = 512

    # Pipeline timing
    silence_timeout: float    = 0.8   # seconds of silence before sending to LLM
    min_prompt_chars: int     = 25
    post_llm_cooldown: float  = 0.0   # prevents feedback loops after LLM response

    # System prompt
    system_prompt: str = (
        "You are a voice assistant named CARAT developed at NIT Rourkela. "
        "Reply briefly."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    speech_end: Optional[float] = None
    llm_request_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_end: Optional[float] = None
    tts_end: Optional[float] = None
    tokens: int = 0

    def reset(self) -> None:
        self.speech_end = None
        self.llm_request_start = None
        self.llm_first_token = None
        self.llm_end = None
        self.tts_end = None
        self.tokens = 0

    def log(self) -> None:
        try:
            asr_to_llm  = self.llm_request_start - self.speech_end       # type: ignore[operator]
            ttft        = self.llm_first_token   - self.llm_request_start # type: ignore[operator]
            llm_dur     = self.llm_end           - self.llm_first_token   # type: ignore[operator]
            tps         = self.tokens / llm_dur if llm_dur > 0 else 0.0
            end_to_end  = self.tts_end           - self.speech_end        # type: ignore[operator]

            logger.info(
                "\n── Timing Metrics ──────────────────────\n"
                "  ASR → LLM decision latency : %.3fs\n"
                "  LLM Time-to-First-Token    : %.3fs\n"
                "  LLM Tokens/sec             : %.2f\n"
                "  End-to-End Latency         : %.3fs\n"
                "────────────────────────────────────────",
                asr_to_llm, ttft, tps, end_to_end,
            )
        except TypeError as exc:
            logger.warning("Metrics incomplete, skipping: %s", exc)


# ---------------------------------------------------------------------------
# TTS (Piper)
# ---------------------------------------------------------------------------

# Matches tokens up to (and including) a sentence-ending punctuation mark.
_CHUNK_RE = re.compile(r"([^.!?]+[.!?]+)")


# class TTSWorker:
#     """Queues text chunks and speaks them sequentially via Piper."""

#     def __init__(self, config: Config) -> None:
#         self._config = config
#         self._queue: queue.Queue[Optional[str]] = queue.Queue()
#         self._thread: Optional[threading.Thread] = None

#     # -- lifecycle -----------------------------------------------------------

#     def start(self) -> None:
#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def stop(self) -> None:
#         """Signal the worker to drain the queue and exit, then join."""
#         self._queue.put(None)
#         if self._thread:
#             self._thread.join()

#     # -- public API ----------------------------------------------------------

#     def enqueue(self, text: str) -> None:
#         self._queue.put(text)

#     # -- internals -----------------------------------------------------------

#     def _run(self) -> None:
#         while True:
#             chunk = self._queue.get()
#             if chunk is None:
#                 break
#             self._speak(chunk)
#             self._queue.task_done()

#     def _speak(self, text: str) -> None:
#         logger.debug("TTS ▶ %s", text[:60])
#         try:
#             piper = subprocess.Popen(
#                 [self._config.piper_bin, "--model", self._config.piper_model, "--output-raw"],
#                 stdin=subprocess.PIPE,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             aplay = subprocess.Popen(
#                 ["aplay", "-r", "16000", "-f", "S16_LE", "-t", "raw", "-"],
#                 stdin=piper.stdout,
#                 stderr=subprocess.DEVNULL,
#             )
#             piper.stdout.close()  # allow piper to receive SIGPIPE when aplay exits
#             _, stderr = piper.communicate(input=text.encode())
#             if stderr:
#                 logger.debug("Piper stderr: %s", stderr.decode().strip())
#             aplay.wait()
#         except Exception:
#             logger.exception("TTS error while speaking chunk")
class TTSWorker:
    """
    Streams text into a single long-lived Piper+aplay pipeline.

    Usage (same API as before):
        tts = TTSWorker(config)
        tts.start()               # call once before LLM loop
        tts.enqueue("Hello.")     # call as tokens arrive
        tts.enqueue("How are you?")
        tts.stop()                # waits for playback to finish
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._piper: Optional[subprocess.Popen] = None
        self._aplay: Optional[subprocess.Popen] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn piper + aplay once, then start the feeder thread."""
        self._piper = subprocess.Popen(
            [
                self._config.piper_bin,
                "--model",       self._config.piper_model,
                "--output-raw",
                "--sentence-silence", "0.0",   # no inter-sentence gap inside piper
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._aplay = subprocess.Popen(
            ["aplay", "-r", "15000", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=self._piper.stdout,
            stderr=subprocess.DEVNULL,
        )
        # Let piper own stdout so aplay can read it directly;
        # closing our copy prevents a deadlock when piper exits.
        self._piper.stdout.close()

        self._thread = threading.Thread(target=self._feeder, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Signal end-of-response, drain the queue, close piper's stdin,
        and wait for aplay to finish playing everything out.
        """
        self._queue.put(None)           # sentinel
        if self._thread:
            self._thread.join()         # wait for feeder to finish writing
        if self._piper and self._piper.stdin:
            try:
                self._piper.stdin.close()   # EOF → piper flushes + exits
            except BrokenPipeError:
                pass
        if self._aplay:
            self._aplay.wait()          # wait for audio to finish playing
        if self._piper:
            self._piper.wait()

    # ── Public API ───────────────────────────────────────────────────────────

    def enqueue(self, text: str) -> None:
        """Queue a text chunk. Safe to call from any thread."""
        self._queue.put(text)

    # ── Internals ────────────────────────────────────────────────────────────

    def _feeder(self) -> None:
        """
        Pull text chunks off the queue and write them to piper's stdin.
        Piper synthesises audio continuously; aplay plays it as it arrives.
        """
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            if not chunk.strip():
                continue
            logger.debug("TTS ▶ %s", chunk[:60])
            try:
                self._piper.stdin.write((chunk + "\n").encode())   # type: ignore[union-attr]
                self._piper.stdin.flush()
            except BrokenPipeError:
                logger.warning("Piper pipe broken — TTS stopping early")
                break



# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the llama.cpp OpenAI-compatible endpoint."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = requests.Session()

    def warmup(self) -> None:
        """Send a single-token request to prime the KV cache."""
        try:
            self._session.post(
                self._config.llama_url,
                json={
                    "messages": [
                        {"role": "system", "content": self._config.system_prompt},
                        {"role": "user",   "content": "hi"},
                    ],
                    "n_predict": 1,
                    "stream": False,
                    "cache_prompt": True,
                },
                timeout=10,
            )
            logger.info("LLM warmup complete")
        except Exception:
            logger.warning("LLM warmup failed — continuing anyway")

    def stream(self, prompt: str):
        """
        Yield decoded token strings from a streaming completion.

        Raises requests.HTTPError on a non-2xx response.
        """
        payload = {
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user",   "content": prompt[-self._config.max_prompt_chars:]},
            ],
            "n_predict":    self._config.n_predict,
            "temperature":  self._config.temperature,
            "cache_prompt": True,
            "stream":       True,
        }
        with self._session.post(
            self._config.llama_url,
            json=payload,
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    return
                event = json.loads(payload_str)
                delta = event["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta


# ---------------------------------------------------------------------------
# Voice Assistant — top-level orchestrator
# ---------------------------------------------------------------------------

class VoiceAssistant:
    """Orchestrates ASR → LLM → TTS with thread-safe state management."""

    def __init__(self, config: Config) -> None:
        self._cfg     = config
        self._llm     = LLMClient(config)
        self._metrics = Metrics()

        # Threading primitives
        self._stop       = threading.Event()
        self._state_lock = threading.Lock()

        # Shared mutable state (always accessed under _state_lock)
        self._listening: bool  = True
        self._llm_busy:  bool  = False
        self._last_llm_time: float = 0.0
        self._last_prompt: str = ""

        # Inter-thread queues
        self._text_queue: queue.Queue[str] = queue.Queue()

    # -- Lifecycle -----------------------------------------------------------

    def run(self) -> None:
        self._llm.warmup()

        whisper_thread = threading.Thread(target=self._asr_loop,  daemon=True, name="ASR")
        agg_thread     = threading.Thread(target=self._agg_loop,  daemon=True, name="Aggregator")

        whisper_thread.start()
        agg_thread.start()

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
            self._stop.set()
            sys.exit(0)

    # -- Properties (thread-safe) -------------------------------------------

    @property
    def listening(self) -> bool:
        with self._state_lock:
            return self._listening

    @property
    def llm_busy(self) -> bool:
        with self._state_lock:
            return self._llm_busy

    # -- ASR loop ------------------------------------------------------------

    def _asr_loop(self) -> None:
        logger.info("ASR thread started — listening…")
        cmd = [
            "taskset", "-c", self._cfg.whisper_cpu_cores,
            self._cfg.whisper_bin,
            "-m",    self._cfg.whisper_model,
            "--step",   str(self._cfg.whisper_step),
            "--length", str(self._cfg.whisper_length),
            "-c",    str(self._cfg.whisper_card),
            "-t",    str(self._cfg.whisper_threads),
            "-ac",   str(self._cfg.whisper_audio_ctx),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=self._cfg.whisper_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:  # type: ignore[union-attr]
            if self._stop.is_set():
                break
            if not self.listening:
                continue
            line = line.strip()
            if not line:
                continue
            # Strip Whisper timestamp prefix, e.g. "[00:00:00.000 --> 00:00:03.000]  text"
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                self._text_queue.put(line)

        proc.terminate()

    # -- Aggregator loop -----------------------------------------------------

    def _agg_loop(self) -> None:
        buffer    = ""
        last_update = time.monotonic()

        while not self._stop.is_set():
            # Drain incoming ASR text
            try:
                text = self._text_queue.get(timeout=0.1)
                if not self.llm_busy and text.strip():
                    buffer += " " + text.strip()
                    last_update = time.monotonic()
                    print(f"\r🎤 {buffer}", end="", flush=True)
            except queue.Empty:
                pass

            if self.llm_busy:
                continue

            # Silence-based commit
            silence = time.monotonic() - last_update
            if buffer and silence > self._cfg.silence_timeout:
                buffer, last_update = self._try_commit(buffer)

    def _try_commit(self, buffer: str) -> tuple[str, float]:
        """Validate and dispatch a buffered prompt; returns a reset (buffer, timestamp)."""
        now = time.monotonic()

        # Cooldown guard
        with self._state_lock:
            since_last = now - self._last_llm_time
        if since_last < self._cfg.post_llm_cooldown:
            return "", now

        prompt = buffer.strip()

        if len(prompt) < self._cfg.min_prompt_chars:
            return "", now

        with self._state_lock:
            if prompt == self._last_prompt:
                return "", now

        logger.info("Sending to LLM: %s", prompt[:80])
        self._run_llm(prompt, speech_end=now)
        return "", time.monotonic()

    # -- LLM + TTS pipeline --------------------------------------------------

    def _run_llm(self, prompt: str, speech_end: float) -> None:
        self._metrics.reset()
        self._metrics.speech_end       = speech_end
        self._metrics.llm_request_start = time.monotonic()

        with self._state_lock:
            self._listening = False
            self._llm_busy  = True
            self._last_prompt = prompt

        tts = TTSWorker(self._cfg)
        tts.start()

        partial = ""

        try:
            for token in self._llm.stream(prompt):
                if self._metrics.llm_first_token is None:
                    self._metrics.llm_first_token = time.monotonic()
                self._metrics.tokens += 1

                print(token, end="", flush=True)
                partial += token

                # Flush complete sentence chunks to TTS immediately
                for match in _CHUNK_RE.findall(partial):
                    chunk = match.strip()
                    if chunk:
                        tts.enqueue(chunk)
                partial = _CHUNK_RE.sub("", partial)

            # Flush any remaining partial sentence
            if partial.strip():
                tts.enqueue(partial.strip())

            print()  # newline after streamed output

        except Exception:
            logger.exception("LLM streaming error")

        finally:
            tts.stop()
            self._metrics.tts_end = time.monotonic()
            self._metrics.llm_end = time.monotonic()

            with self._state_lock:
                self._llm_busy  = False
                self._listening = True
                self._last_llm_time = time.monotonic()

            logger.info("Listening…")
            self._metrics.log()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    VoiceAssistant(Config()).run()


if __name__ == "__main__":
    main()
