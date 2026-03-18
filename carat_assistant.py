"""
CARAT Voice Assistant
Developed at NIT Rourkela

Pipeline: OpenWakeWord → Whisper ASR → LLaMA LLM (streaming) → Piper TTS

Usage:
    python carat_assistant.py            # English (default)
    python carat_assistant.py -l hi      # Hindi
    python carat_assistant.py --language en
"""

import argparse
import json
import logging
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyaudio
import requests
from openwakeword.model import Model as WakeWordModel

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
# System prompts per language
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "en": (
        "You are a voice assistant named CARAT developed at NIT Rourkela. "
        "Reply briefly."
    ),
    "hi": (
        "आप CARAT नाम के एक वॉयस असिस्टेंट हैं जिसे NIT राउरकेला में विकसित किया गया है। "
        "संक्षेप में उत्तर दें।"
    ),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    # Piper TTS
    piper_bin: str   = "./piper/piper"
    piper_model: str = "/home/robot/Desktop/trial/llm_pi/en_US-ryan-low.onnx"

    # Whisper ASR
    whisper_bin: str    = "./build/bin/whisper-stream"
    whisper_model: str  = "./models/ggml-base.en.bin"
    whisper_cwd: str    = "../whisper.cpp"
    whisper_step: int   = 4000
    whisper_length: int = 8000
    whisper_threads: int = 3
    whisper_audio_ctx: int = 512

    # LLaMA
    llama_url: str     = "http://localhost:8080/v1/chat/completions"
    n_predict: int     = 40
    temperature: float = 0.4
    max_prompt_chars: int = 512

    # Pipeline timing
    silence_timeout: float   = 1.2   # seconds of silence before sending to LLM
    min_prompt_chars: int    = 15
    post_llm_cooldown: float = 3.0   # prevents feedback loops after LLM response

    # Wake word
    wakeword_model_paths: list = None   # None → openwakeword uses its built-in models
    wakeword_threshold: float  = 0.5
    wakeword_chunk_size: int   = 1280   # samples per chunk at 16 kHz (80 ms)
    wakeword_name: str         = "hey_jarvis"  # model key to watch; adjust to your model

    def __init__(self, language: str = "en") -> None:
        self.language      = language
        self.system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
        if self.wakeword_model_paths is None:
            self.wakeword_model_paths = []


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
        self.speech_end        = None
        self.llm_request_start = None
        self.llm_first_token   = None
        self.llm_end           = None
        self.tts_end           = None
        self.tokens            = 0

    def log(self) -> None:
        try:
            asr_to_llm = self.llm_request_start - self.speech_end        # type: ignore[operator]
            ttft       = self.llm_first_token   - self.llm_request_start # type: ignore[operator]
            llm_dur    = self.llm_end           - self.llm_first_token   # type: ignore[operator]
            tps        = self.tokens / llm_dur if llm_dur > 0 else 0.0
            end_to_end = self.tts_end           - self.speech_end        # type: ignore[operator]

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
# Wake Word Detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """
    Listens on the microphone and sets `wake_event` when the wake word fires.

    The detector runs in its own daemon thread.  Call `start()` once.
    The aggregator loop waits on `wake_event`; after consuming the event it
    calls `reset()` to arm the detector for the next utterance.
    """

    def __init__(self, config: Config, wake_event: threading.Event) -> None:
        self._cfg        = config
        self._wake_event = wake_event
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Initialise openwakeword; pass explicit model paths only if provided
        oww_kwargs = {}
        if config.wakeword_model_paths:
            oww_kwargs["wakeword_models"] = config.wakeword_model_paths
        self._model = WakeWordModel(**oww_kwargs)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="WakeWord")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        """Re-arm: clear the event so the detector can fire again."""
        self._wake_event.clear()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._cfg.wakeword_chunk_size,
        )
        logger.info("Wake-word detector ready — say the wake word to activate.")

        try:
            while not self._stop.is_set():
                # Skip processing while wake event is already set (assistant active)
                if self._wake_event.is_set():
                    time.sleep(0.05)
                    continue

                raw = stream.read(self._cfg.wakeword_chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(raw, dtype=np.int16)

                # openwakeword expects float32 in [-1, 1]
                audio_f32 = audio_np.astype(np.float32) / 32768.0
                self._model.predict(audio_f32)

                scores = self._model.prediction_buffer
                # scores is a dict: {model_name: deque_of_scores}
                for model_name, score_buffer in scores.items():
                    if not score_buffer:
                        continue
                    latest = score_buffer[-1]
                    if latest >= self._cfg.wakeword_threshold:
                        logger.info(
                            "Wake word detected! (model=%s, score=%.3f)",
                            model_name, latest,
                        )
                        print("\n🔔 Wake word detected — listening…", flush=True)
                        self._wake_event.set()
                        break
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


# ---------------------------------------------------------------------------
# TTS (Piper)
# ---------------------------------------------------------------------------

_CHUNK_RE = re.compile(r"([^.!?]+[.!?]+)")


class TTSWorker:
    """
    Streams text into a single long-lived Piper + aplay pipeline.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._piper: Optional[subprocess.Popen] = None
        self._aplay: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._piper = subprocess.Popen(
            [
                self._config.piper_bin,
                "--model",      self._config.piper_model,
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._aplay = subprocess.Popen(
            ["paplay", "--rate", "15000", "--format", "s16le", "--channels=1", "--raw"],
            stdin=self._piper.stdout,
            stderr=subprocess.DEVNULL,
        )
        self._piper.stdout.close()
        self._thread = threading.Thread(target=self._feeder, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join()
        if self._piper and self._piper.stdin:
            try:
                self._piper.stdin.close()
            except BrokenPipeError:
                pass
        if self._aplay:
            self._aplay.wait()
        if self._piper:
            self._piper.wait()

    def enqueue(self, text: str) -> None:
        self._queue.put(text)

    def _feeder(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            if not chunk.strip():
                continue
            logger.debug("TTS ▶ %s", chunk[:60])
            try:
                self._piper.stdin.write((chunk + "\n").encode())  # type: ignore[union-attr]
                self._piper.stdin.flush()
            except BrokenPipeError:
                logger.warning("Piper pipe broken — TTS stopping early")
                break


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, config: Config) -> None:
        self._config  = config
        self._session = requests.Session()

    def warmup(self) -> None:
        try:
            self._session.post(
                self._config.llama_url,
                json={
                    "messages": [
                        {"role": "system", "content": self._config.system_prompt},
                        {"role": "user",   "content": "hi"},
                    ],
                    "n_predict":    1,
                    "stream":       False,
                    "cache_prompt": True,
                },
                timeout=10,
            )
            logger.info("LLM warmup complete")
        except Exception:
            logger.warning("LLM warmup failed — continuing anyway")

    def stream(self, prompt: str):
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
    def __init__(self, config: Config) -> None:
        self._cfg     = config
        self._llm     = LLMClient(config)
        self._metrics = Metrics()

        self._stop       = threading.Event()
        self._state_lock = threading.Lock()

        # Wake-word gate: ASR aggregator only commits prompts when this is set
        self._wake_event = threading.Event()

        self._listening: bool  = True
        self._llm_busy:  bool  = False
        self._last_llm_time: float = 0.0
        self._last_prompt: str = ""

        self._text_queue: queue.Queue[str] = queue.Queue()

        # Wake-word detector
        self._wakeword = WakeWordDetector(config, self._wake_event)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._llm.warmup()
        self._wakeword.start()

        whisper_thread = threading.Thread(target=self._asr_loop, daemon=True, name="ASR")
        agg_thread     = threading.Thread(target=self._agg_loop, daemon=True, name="Aggregator")

        whisper_thread.start()
        agg_thread.start()

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
            self._stop.set()
            self._wakeword.stop()
            sys.exit(0)

    # ── Thread-safe properties ───────────────────────────────────────────────

    @property
    def listening(self) -> bool:
        with self._state_lock:
            return self._listening

    @property
    def llm_busy(self) -> bool:
        with self._state_lock:
            return self._llm_busy

    # ── ASR loop ─────────────────────────────────────────────────────────────

    def _asr_loop(self) -> None:
        logger.info("ASR thread started (language=%s) — waiting for wake word…", self._cfg.language)
        cmd = [
            self._cfg.whisper_bin,
            "-m",        self._cfg.whisper_model,
            "--step",    str(self._cfg.whisper_step),
            "--length",  str(self._cfg.whisper_length),
            "-t",        str(self._cfg.whisper_threads),
            "-ac",       str(self._cfg.whisper_audio_ctx),
            "--language", self._cfg.language,   # ← language flag passed here
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
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                self._text_queue.put(line)

        proc.terminate()

    # ── Aggregator loop ───────────────────────────────────────────────────────

    def _agg_loop(self) -> None:
        buffer      = ""
        last_update = time.monotonic()

        while not self._stop.is_set():
            # ── Gate: wait for wake word ──────────────────────────────────
            if not self._wake_event.is_set():
                # Drain any stale ASR output accumulated while sleeping
                while not self._text_queue.empty():
                    try:
                        self._text_queue.get_nowait()
                    except queue.Empty:
                        break
                buffer = ""
                last_update = time.monotonic()
                time.sleep(0.1)
                continue

            # Drain incoming ASR text
            try:
                text = self._text_queue.get(timeout=0.1)
                if not self.llm_busy and text.strip():
                    buffer     += " " + text.strip()
                    last_update = time.monotonic()
                    print(f"\r🎤 {buffer}", end="", flush=True)
            except queue.Empty:
                pass

            if self.llm_busy:
                buffer      = ""
                last_update = time.monotonic()
                continue

            silence = time.monotonic() - last_update
            if buffer and silence > self._cfg.silence_timeout:
                buffer, last_update = self._try_commit(buffer)

    def _try_commit(self, buffer: str) -> Tuple[str, float]:
        now = time.monotonic()

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

    # ── LLM + TTS pipeline ───────────────────────────────────────────────────

    def _run_llm(self, prompt: str, speech_end: float) -> None:
        self._metrics.reset()
        self._metrics.speech_end        = speech_end
        self._metrics.llm_request_start = time.monotonic()

        with self._state_lock:
            self._listening   = False
            self._llm_busy    = True
            self._last_prompt = prompt

        tts     = TTSWorker(self._cfg)
        partial = ""
        tts.start()

        try:
            for token in self._llm.stream(prompt):
                if self._metrics.llm_first_token is None:
                    self._metrics.llm_first_token = time.monotonic()
                self._metrics.tokens += 1

                print(token, end="", flush=True)
                partial += token

                for match in _CHUNK_RE.findall(partial):
                    chunk = match.strip()
                    if chunk:
                        tts.enqueue(chunk)
                partial = _CHUNK_RE.sub("", partial)

            if partial.strip():
                tts.enqueue(partial.strip())
            print()

        except Exception:
            logger.exception("LLM streaming error")

        finally:
            tts.stop()
            self._metrics.tts_end = time.monotonic()
            self._metrics.llm_end = time.monotonic()

            # Drain stale ASR tokens
            while not self._text_queue.empty():
                try:
                    self._text_queue.get_nowait()
                except queue.Empty:
                    break

            with self._state_lock:
                self._llm_busy      = False
                self._listening     = True
                self._last_llm_time = time.monotonic()

            # Go back to sleep — require another wake word
            self._wakeword.reset()
            logger.info("Response done. Waiting for wake word…")
            self._metrics.log()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARAT Voice Assistant — NIT Rourkela",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--language",
        choices=list(SYSTEM_PROMPTS.keys()),
        default="en",
        help="Whisper transcription language (and system-prompt language). "
             "Supported: en, hi.",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = Config(language=args.language)
    logger.info(
        "Starting CARAT | language=%s | system_prompt=%s…",
        config.language, config.system_prompt[:60],
    )
    VoiceAssistant(config).run()


if __name__ == "__main__":
    main()
