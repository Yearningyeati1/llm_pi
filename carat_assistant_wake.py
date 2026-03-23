"""
CARAT Voice Assistant
Developed at NIT Rourkela

Pipeline: OpenWakeWord → Whisper ASR (PCM via stdin) → LLaMA LLM (streaming) → Piper TTS

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

class Config:
    # Piper TTS
    piper_bin: str = "./piper/piper"

    piper_models: dict = {
        "en": "/home/robot/Desktop/trial/llm_pi/en_US-ryan-low.onnx",
        "hi": "/home/robot/Desktop/trial/llm_pi/hi_IN-pratham-medium.onnx",
    }
    tts_sample_rates: dict = {
        "en": 16000,
        "hi": 22050,
    }

    # Hallucination filter — lowercased, stripped
    asr_hallucinations: set = {
        "thank you", "thanks for watching", "thanks for listening",
        "i'm going to go to the next one.",
        "you", "the", ".", "..", "...", " ", "bye", "bye bye",
        "subtitles by", "subscribe", "please subscribe",
        "[music]", "[silence]", "(silence)", "*", "* *", "* * *", "झाल",
    }

    # Whisper ASR — PCM stdin variant
    whisper_bin: str      = "./build/bin/whisper-stream-pcm"
    whisper_model: str    = "./models/ggml-large-v3-turbo-q5_0.bin"
    whisper_cwd: str     = "../whisper.cpp.pcm"
    whisper_step: int     = 1000
    whisper_length: int   = 3000
    whisper_threads: int  = 2
    whisper_audio_ctx: int = 512
    whisper_vad: bool     = True          # pass --vad flag
    # vth intentionally omitted here — uncomment in _build_whisper_cmd if needed
    # vth: float          = 0.7

    # LLaMA
    llama_url: str        = "http://localhost:8080/v1/chat/completions"
    n_predict: int        = 50
    temperature: float    = 0.4
    max_prompt_chars: int = 512

    # Mic / audio
    mic_device: int       = 24            # PyAudio device index
    mic_rate: int         = 16000
    mic_chunk: int        = 1280          # 80 ms at 16 kHz

    # Wake word
    wakeword_model_paths: list          = ["/home/robot/Desktop/trial/llm_pi/hey_khaa_rat.onnx"]
    wakeword_name: str                  = "hey_khaa_rat"
    wakeword_threshold: float           = 0.001
    wakeword_debounce: float            = 0.0   # seconds between triggers

    # Silence / commit tuning (kept for min-prompt guard)
    min_prompt_chars: int    = 4
    post_llm_cooldown: float = 0.5       # brief pause before re-arming wake word

    def __init__(self, language: str = "en") -> None:
        self.language        = language
        self.system_prompt   = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
        self.piper_model     = self.piper_models.get(language, self.piper_models["en"])
        self.tts_sample_rate = self.tts_sample_rates.get(language, 16000)


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
            asr_to_llm = self.llm_request_start - self.speech_end         # type: ignore[operator]
            ttft       = self.llm_first_token   - self.llm_request_start  # type: ignore[operator]
            llm_dur    = self.llm_end           - self.llm_first_token    # type: ignore[operator]
            tps        = self.tokens / llm_dur if llm_dur > 0 else 0.0
            end_to_end = self.tts_end           - self.speech_end         # type: ignore[operator]

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
            ["paplay", "--rate", str(self._config.tts_sample_rate),
             "--format", "s16le", "--channels=1", "--raw"],
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
# Shared Mic Reader
# ---------------------------------------------------------------------------

class MicReader:
    """Single PyAudio stream at 16 kHz float32 — fans audio to subscribers."""

    RATE  = 16000
    CHUNK = 1280   # 80 ms

    def __init__(self, device_index: int) -> None:
        self._device = device_index
        self._subs: list[queue.Queue] = []
        self._stop = threading.Event()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        self._subs.append(q)
        return q

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="MicReader").start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            #input_device_index=self._device,
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        logger.info("MicReader: device=%d @ %d Hz float32", self._device, self.RATE)
        try:
            while not self._stop.is_set():
                raw   = stream.read(self.CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(raw, dtype=np.float32)
                for q in self._subs:
                    try:
                        q.put_nowait(audio)
                    except queue.Full:
                        pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

# ---------------------------------------------------------------------------
# Wake Word Detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    def __init__(self, config: Config, wake_event: threading.Event,
                 audio_queue: queue.Queue) -> None:
        self._cfg        = config
        self._wake_event = wake_event
        self._audio_q    = audio_queue
        self._stop       = threading.Event()

        if config.wakeword_model_paths:
            self._model = WakeWordModel(
                wakeword_models=config.wakeword_model_paths,
                inference_framework="onnx",
            )
        else:
            self._model = WakeWordModel(inference_framework="onnx")

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="WakeWord").start()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        """Clear model buffers and re-arm the detector."""
        for buf in self._model.prediction_buffer.values():
            buf.clear()
        self._wake_event.clear()
        logger.info("Wake-word detector re-armed — say the wake word to activate.")
        print("💤 Waiting for wake word…\n", flush=True)

    def _run(self) -> None:
        logger.info("Wake-word detector ready — say the wake word to activate.")
        last_trigger = 0.0

        while not self._stop.is_set():
            try:
                audio_16k = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Don't evaluate while already awake
            if self._wake_event.is_set():
                continue
            
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            self._model.predict(audio_int16)

            for model_name, score_buffer in self._model.prediction_buffer.items():
                if self._cfg.wakeword_name.lower() not in model_name.lower():
                    continue
                if not score_buffer:
                    continue
                if score_buffer[-1] >= self._cfg.wakeword_threshold:
                    now = time.time()
                    if now - last_trigger > self._cfg.wakeword_debounce:
                        logger.info("Wake word detected! (model=%s, score=%.3f)",
                                    model_name, score_buffer[-1])
                        print("\n🔔 Wake word detected — listening…", flush=True)
                        self._play_beep()
                        self._wake_event.set()
                        last_trigger = now
                    break

    # Add this method to WakeWordDetector class
    def _play_beep(self) -> None:
        """Play a short confirmation beep via paplay."""
        sample_rate = 22050
        duration    = 0.15   # seconds
        frequency   = 1000    # Hz (A5 — a clear, pleasant ping)
    
        t     = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave  = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Fade out to avoid a click at the end
        fade  = np.linspace(1.0, 0.0, len(wave))
        wave  = (wave * fade).astype(np.int16)
    
        proc = subprocess.Popen(
        ["paplay", "--rate", str(sample_rate),
         "--format", "s16le", "--channels=1", "--raw"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(wave.tobytes())
        proc.stdin.close()
        proc.wait()


# ---------------------------------------------------------------------------
# Whisper Streamer  (PCM via stdin, VAD-gated, wake-word aware)
# ---------------------------------------------------------------------------

class WhisperStreamer:
    """
    Feeds raw PCM from the shared MicReader into whisper-stream-pcm only
    after the wake event fires.  Collects the full utterance between the
    ### Transcription START / END markers, applies hallucination filtering,
    then calls back into the VoiceAssistant to trigger LLM + TTS.
    """

    def __init__(self, config: Config, audio_queue: queue.Queue,
                 wake_event: threading.Event,
                 on_transcript,           # callable(str) → None
                 wakeword: WakeWordDetector) -> None:
        self._cfg        = config
        self._audio_q    = audio_queue
        self._wake_event = wake_event
        self._on_transcript = on_transcript
        self._wakeword   = wakeword
        self._stop       = threading.Event()

        self._proc: Optional[subprocess.Popen] = None
        self._active_session = False
        self._transcript_taken = False  

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        cmd = [
            self._cfg.whisper_bin,
            "-m",            self._cfg.whisper_model,
            "--language",    self._cfg.language,
            "--step",        str(self._cfg.whisper_step),
            "--length",      str(self._cfg.whisper_length),
            "-t",            str(self._cfg.whisper_threads),
            "-ac",           str(self._cfg.whisper_audio_ctx),
            "-i",            "-",
            "--format",      "f32",
            "--sample-rate", str(self._cfg.mic_rate),
        ]
        if self._cfg.whisper_vad:
            cmd.append("--vad")
        # Uncomment to set VAD threshold:
        # cmd += ["-vth", str(self._cfg.vth)]

        self._proc = subprocess.Popen(
            cmd,
            cwd=self._cfg.whisper_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            text=False,
        )
        logger.info("Whisper process started (PCM stdin mode)")

        threading.Thread(target=self._feed_audio, daemon=True, name="WhisperFeed").start()
        threading.Thread(target=self._read_output, daemon=True, name="WhisperRead").start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    # ── Audio feeder ─────────────────────────────────────────────────────────

    def _feed_audio(self) -> None:
        """
        Convert int16 chunks (from MicReader) to float32 and write to
        Whisper stdin.  Audio is only forwarded while the wake event is set.
        """
        while not self._stop.is_set():
            try:
                audio = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._wake_event.is_set():
                self._active_session = False
                continue

            if not self._active_session:
                logger.info("Whisper session started — VAD listening…")
                print("🧠 Listening (VAD)…", flush=True)
                self._active_session = True
                self._transcript_taken = False

            try:
                self._proc.stdin.write(audio.tobytes())  # type: ignore[union-attr]
            except (BrokenPipeError, AttributeError):
                break

    # ── Output parser ────────────────────────────────────────────────────────
    def _drain_whisper_stdin(self) -> None:
        if not self._proc or not self._proc.stdin:
            return
        silence = np.zeros(self._cfg.mic_rate, dtype=np.float32)
        try:
            self._proc.stdin.write(silence.tobytes())
        except (BrokenPipeError, AttributeError):
            pass
        logger.debug("Whisper stdin drained with silence.")


    def _read_output(self) -> None:
        collecting = False
        buffer: list[str] = []

        for raw in self._proc.stdout:  # type: ignore[union-attr]
            if self._stop.is_set():
                break

            line = raw.decode(errors="ignore").strip()
            if not line:
                continue

            # ── START marker ────────────────────────────────────────────────
            if "### Transcription" in line and "START" in line:
                if self._transcript_taken:
                    logger.debug("Ignoring extra START block (transcript already taken)")
                    collecting = False
                else:
                    collecting = True
                    buffer = []
                    continue

            # ── END marker ──────────────────────────────────────────────────
            if "### Transcription" in line and "END" in line:
                if not collecting:
                # This END belongs to a skipped block — drain and ignore
                    self._drain_whisper_stdin()
                    continue

                collecting = False
                speech_end = time.monotonic()
                final_text = " ".join(buffer).strip()

                if final_text:
                    self._transcript_taken = True          # ← lock out further blocks
                    self._drain_whisper_stdin()            # ← flush stale audio
                    print(f"\n📝 TRANSCRIPT: {final_text}", flush=True)
                    self._on_transcript(final_text, speech_end)
                else:
                    logger.debug("Empty transcript after END marker — re-arming.")
                    self._active_session = False
                    self._transcript_taken = False
                    self._wakeword.reset()

                continue

            # ── Text lines ──────────────────────────────────────────────────
            if collecting:
                # Strip timestamp prefix like "[00:00:00.000 --> 00:00:02.000]  text"
                text = line.split("]", 1)[-1].strip() if "]" in line else line

                if not text:
                    continue

                # Hallucination filter
                clean = text.lower().strip(".,!? ")
                if clean in self._cfg.asr_hallucinations:
                    logger.debug("Filtered hallucination: %r", text)
                    continue

                buffer.append(text)

        logger.warning("Whisper stdout closed.")


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

        self._llm_busy: bool      = False
        self._last_prompt: str    = ""

        # Shared mic — two subscribers: one for wake word, one for Whisper
        self._mic = MicReader(config.mic_device)
        ww_queue      = self._mic.subscribe()
        whisper_queue = self._mic.subscribe()

        self._wakeword = WakeWordDetector(config, self._wake_event, ww_queue)

        self._whisper  = WhisperStreamer(
            config       = config,
            audio_queue  = whisper_queue,
            wake_event   = self._wake_event,
            on_transcript= self._on_transcript,
            wakeword     = self._wakeword,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._llm.warmup()
        self._mic.start()
        self._wakeword.start()
        self._whisper.start()
        print("💤 Waiting for wake word…\n", flush=True)

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down…")
            self._stop.set()
            self._mic.stop()
            self._whisper.stop()
            sys.exit(0)

    # ── Transcript callback (called from WhisperRead thread) ─────────────────

    def _on_transcript(self, text: str, speech_end: float) -> None:
        """
        Called when Whisper delivers a complete, filtered utterance.
        Runs LLM + TTS synchronously in the WhisperRead thread so that
        audio feeding is naturally paused during the response (wake event
        is still set but _active_session blocks re-entry).
        """
        with self._state_lock:
            if self._llm_busy:
                logger.info("LLM busy — ignoring transcript: %r", text[:60])
                self._wakeword.reset()
                return

            if len(text) < self._cfg.min_prompt_chars:
                logger.info("Prompt too short (%d chars) — ignoring.", len(text))
                self._wakeword.reset()
                return

            if text == self._last_prompt:
                logger.info("Duplicate prompt — ignoring.")
                self._wakeword.reset()
                return

            self._llm_busy    = True
            self._last_prompt = text

        self._run_llm(text, speech_end)

    # ── LLM + TTS pipeline ───────────────────────────────────────────────────

    def _run_llm(self, prompt: str, speech_end: float) -> None:
        self._metrics.reset()
        self._metrics.speech_end        = speech_end
        self._metrics.llm_request_start = time.monotonic()

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

            # Brief cooldown so mic doesn't immediately pick up TTS audio
            time.sleep(self._cfg.post_llm_cooldown)

            with self._state_lock:
                self._llm_busy = False

            # Re-arm wake-word detector for next cycle
            self._wakeword.reset()

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
