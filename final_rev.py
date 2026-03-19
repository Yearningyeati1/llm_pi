"""
CARAT Voice Assistant
Developed at NIT Rourkela

Pipeline: OpenWakeWord → Whisper ASR → LLaMA LLM (streaming) → Piper TTS

Usage:
    python carat_assistant.py            # English (default)
    python carat_assistant.py -l hi      # Hindi
    python carat_assistant.py --language en
"""

# ── Suppress ALSA warnings ────────────────────────────────────────────────────
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from ctypes import cdll, c_char_p, c_int, CFUNCTYPE

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass  # silently swallow ALSA errors

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so.2')
asound.snd_lib_error_set_handler(c_error_handler)
# ─────────────────────────────────────────────────────────────────────────────
 
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

# --------------------------------------------------------------------------
class Config:
    # Piper TTS
    piper_bin: str = "./piper/piper"

    # Model path per language — add new languages here as needed
    piper_models: dict = {
        "en": "/home/robot/Desktop/trial/llm_pi/en_US-ryan-low.onnx",
        "hi": "/home/robot/Desktop/trial/llm_pi/hi_IN-pratham-medium.onnx",
    }
    tts_sample_rates: dict = {
    "en": 16000,
    "hi": 22050,
    }
    asr_hallucinations: set = {
    "thank you", "thanks for watching", "thanks for listening","I'm going to go to the next one."
    "you", "the", ".", "..", "...", " ", "bye", "bye bye",
    "subtitles by", "subscribe", "please subscribe",
    "[music]", "[silence]", "(silence)", "*", "* *", "* * *","झाल"
    }

    # Whisper ASR
    whisper_bin: str     = "./build/bin/whisper-stream"
    whisper_model: str   = "./models/ggml-large-v3-turbo-q5_0.bin"
    whisper_cwd: str     = "../whisper.cpp"
    whisper_step: int    = 4000
    whisper_length: int  = 6000
    whisper_threads: int = 3
    whisper_audio_ctx: int = 512
    vth: float             = 0.7
    capture_device: int    = 1

    # LLaMA
    llama_url: str        = "http://localhost:8080/v1/chat/completions"
    n_predict: int        = 50
    temperature: float    = 0.4
    max_prompt_chars: int = 512

    # Pipeline timing
    silence_timeout: float   = 1.2   # seconds of silence before sending to LLM
    min_prompt_chars: int    = 25
    post_llm_cooldown: float = 3.0   # prevents feedback loops after LLM response
    
    # Mic device
    mic_device: int = 24

    # Wake word
    wakeword_model_paths: list         = ["/home/robot/Desktop/trial/llm_pi/hey_khaa_rat.onnx"]
    wakeword_name: str                 = "hey_khaa_rat"
    wakeword_threshold: float          = 0.01
    wakeword_chunk_size: int           = 1280   # samples per chunk at 16 kHz (80 ms)
    wakeword_input_device: Optional[int] = 24   # set to None to use system default mic

    def __init__(self, language: str = "en") -> None:
        self.language      = language
        self.system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
        self.piper_model   = self.piper_models.get(language, self.piper_models["en"])
        self.tts_sample_rate = self.tts_sample_rates.get(language, 16000)
        if self.wakeword_model_paths is None:
            self.wakeword_model_paths = []



# ---------------------------------------------------------------------------
# Shared Mic Reader
# ---------------------------------------------------------------------------

class MicReader:
    """Single PyAudio stream on hw:3,0 — fans raw audio to subscribers."""

    TARGET_RATE = 16000
    CHUNK       = 1280  # 80ms at 16kHz

    def __init__(self, device_index: int) -> None:
        self._device = device_index
        self._subs: list[queue.Queue] = []
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        self._subs.append(q)
        return q

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="MicReader")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        pa       = pyaudio.PyAudio()
        dev_info = pa.get_device_info_by_index(self._device)
        NATIVE_RATE  = int(dev_info['defaultSampleRate'])
        NATIVE_CHUNK = int(self.CHUNK * NATIVE_RATE / self.TARGET_RATE)

        logger.info("MicReader: device=[%s] native_rate=%d → resampling to %d Hz",
                    dev_info['name'], NATIVE_RATE, self.TARGET_RATE)

        stream = pa.open(
            input_device_index=self._device,
            rate=NATIVE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=NATIVE_CHUNK,
        )
        try:
            while not self._stop.is_set():
                raw      = stream.read(NATIVE_CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(raw, dtype=np.int16)
                audio_16k = resample_audio(audio_np, NATIVE_RATE, self.TARGET_RATE)
                for q in self._subs:
                    try:
                        q.put_nowait(audio_16k)
                    except queue.Full:
                        pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


# ---------------------------------------------------------------------------
# Wake Word Detector  (now reads from queue)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    def __init__(self, config: Config, wake_event: threading.Event,
                 audio_queue: queue.Queue) -> None:
        self._cfg        = config
        self._wake_event = wake_event
        self._audio_q    = audio_queue
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if config.wakeword_model_paths:
            self._model = WakeWordModel(
                wakeword_models=config.wakeword_model_paths,
                inference_framework="onnx",
            )
        else:
            self._model = WakeWordModel(inference_framework="onnx")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="WakeWord")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        for buf in self._model.prediction_buffer.values():
            buf.clear()
        self._wake_event.clear()

    def _run(self) -> None:
        logger.info("Wake-word detector ready — say the wake word to activate.")
        while not self._stop.is_set():
            try:
                audio_16k = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._wake_event.is_set():
                continue

            self._model.predict(audio_16k)

            for model_name, score_buffer in self._model.prediction_buffer.items():
                if self._cfg.wakeword_name.lower() not in model_name.lower():
                    continue
                if not score_buffer:
                    continue
                if score_buffer[-1] >= self._cfg.wakeword_threshold:
                    logger.info("Wake word detected! (model=%s, score=%.3f)",
                                model_name, score_buffer[-1])
                    print("\n🔔 Wake word detected — listening…", flush=True)
                    self._wake_event.set()
                    break
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
        "Reply briefly. Don't use special symbols."
    ),
    "hi": (
        "आप CARAT नाम के एक वॉयस असिस्टेंट हैं जिसे NIT राउरकेला में विकसित किया गया है। "
        "संक्षेप में उत्तर दें।"
    ),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _find_device_index(pa, name_fragment):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name_fragment.lower() in info['name'].lower() and info['maxInputChannels'] > 0:
            return i
    return None



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_audio(audio_np: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample — sufficient quality for wake-word / speech detection."""
    if orig_rate == target_rate:
        return audio_np
    new_length = int(len(audio_np) * target_rate / orig_rate)
    resampled  = np.interp(
        np.linspace(0, len(audio_np) - 1, new_length),
        np.arange(len(audio_np)),
        audio_np.astype(np.float32),
    )
    return resampled.astype(np.int16)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    speech_end: Optional[float]        = None
    llm_request_start: Optional[float] = None
    llm_first_token: Optional[float]   = None
    llm_end: Optional[float]           = None
    tts_end: Optional[float]           = None
    tokens: int                        = 0

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
# TTS (Piper)
# ---------------------------------------------------------------------------

_CHUNK_RE = re.compile(r"([^.!?]+[.!?]+)")


class TTSWorker:
    """Streams text into a single long-lived Piper + paplay pipeline."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._piper: Optional[subprocess.Popen]  = None
        self._aplay: Optional[subprocess.Popen]  = None

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
            ["paplay", "--rate", str(self._config.tts_sample_rate), "--format", "s16le", "--channels=1", "--raw"],
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

        self._wake_event = threading.Event()

        self._listening: bool      = True
        self._llm_busy:  bool      = False
        self._last_llm_time: float = 0.0
        self._last_prompt: str     = ""
        self._mic = MicReader(config.mic_device)
        ww_queue      = self._mic.subscribe()

        self._text_queue: queue.Queue[str] = queue.Queue()

        self._wakeword = WakeWordDetector(config, self._wake_event, ww_queue)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._llm.warmup()
        self._mic.start()
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
            self._mic.stop()
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
            "-m",         self._cfg.whisper_model,
            "--step",     str(self._cfg.whisper_step),
            "--length",   str(self._cfg.whisper_length),
            "-t",         str(self._cfg.whisper_threads),
            "-ac",        str(self._cfg.whisper_audio_ctx),
            "--language", self._cfg.language,
            "-vth",       str(self._cfg.vth),
            "-c",         "1"#str(self._cfg.capture_device)
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
            if not self.listening or not self._wake_event.is_set():
                continue
            line = line.strip()
            if not line:
                continue
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                self._text_queue.put(line)
            # ── Hallucination filters ──────────────────────────────────────
            clean = line.lower().strip(".,!? ")
            if clean in self._cfg.asr_hallucinations:
                logger.debug("Filtered hallucination: %r", line)
                continue

        proc.terminate()

    # ── Aggregator loop ───────────────────────────────────────────────────────

    def _agg_loop(self) -> None:
        buffer      = ""
        last_update = time.monotonic()
        awake       = False

        while not self._stop.is_set():
            if not self._wake_event.is_set():
                if awake:
                    buffer      = ""
                    last_update = time.monotonic()
                    awake       = False
                time.sleep(0.05)
                continue

            if not awake:
                buffer      = ""
                last_update = time.monotonic()
                awake       = True

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

            # Clear wake event first so ASR gate closes immediately
            self._wakeword.reset()

            # Drain anything that slipped in before the gate closed
            drained = 0
            while not self._text_queue.empty():
                try:
                    self._text_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained:
                logger.debug("Drained %d stale ASR tokens after response", drained)

            with self._state_lock:
                self._llm_busy      = False
                self._listening     = True
                self._last_llm_time = time.monotonic()

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
