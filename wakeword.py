"""
CARAT Voice Assistant
Developed at NIT Rourkela

Pipeline: Porcupine Wake Word → Whisper ASR → LLaMA LLM (streaming) → Piper TTS
"""

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

import pvporcupine
import pyaudio
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
    silence_timeout: float   = 1.2
    min_prompt_chars: int    = 15
    post_llm_cooldown: float = 3.0

    # Wake word (Porcupine)
    porcupine_access_key: str = "YOUR_PICOVOICE_ACCESS_KEY"
    # Use a built-in keyword or path to a .ppn file:
    #   Built-ins: "hey porcupine", "alexa", "ok google", "hey siri", etc.
    #   Custom:    "/path/to/carat_wakeword.ppn"
    wake_word: str = "hey porcupine"
    # Seconds of audio to discard after wake-word detection so the keyword
    # itself is not transcribed and sent to the LLM.
    wake_word_discard_secs: float = 0.6

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
            asr_to_llm = self.llm_request_start - self.speech_end        # type: ignore[operator]
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
    """Streams text into a single long-lived Piper + aplay pipeline."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._piper: Optional[subprocess.Popen] = None
        self._aplay: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._piper = subprocess.Popen(
            [self._config.piper_bin, "--model", self._config.piper_model, "--output-raw"],
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
    """Thin wrapper around the llama.cpp OpenAI-compatible endpoint."""

    def __init__(self, config: Config) -> None:
        self._config = config
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
# Wake Word Detector (Porcupine)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """
    Runs Porcupine in its own thread, reading directly from the microphone
    via PyAudio.  Fires a threading.Event when the wake word is heard.

    Porcupine requires frames of exactly `frame_length` samples at 16 kHz,
    16-bit mono — we handle all of that here so the rest of the pipeline
    is unaffected.

    Lifecycle
    ---------
        detector = WakeWordDetector(config, wake_event)
        detector.start()   # call once at startup
        ...
        detector.pause()   # while ASR / LLM / TTS is running
        detector.resume()  # when ready to listen for wake word again
        detector.stop()    # clean shutdown
    """

    def __init__(self, config: Config, wake_event: threading.Event) -> None:
        self._config     = config
        self._wake_event = wake_event
        self._paused     = threading.Event()   # set → detector is paused
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="WakeWord")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()          # unblock if waiting
        if self._thread:
            self._thread.join(timeout=3)

    def pause(self) -> None:
        """Call before starting ASR/LLM/TTS to suppress detection."""
        self._paused.set()

    def resume(self) -> None:
        """Call after TTS finishes to re-enable detection."""
        self._paused.clear()
        logger.info("🎙  Waiting for wake word…")

    # ── Detection loop ────────────────────────────────────────────────────

    def _run(self) -> None:
        # Choose keyword source: built-in string or path to .ppn file
        kw = self._config.wake_word
        if kw.endswith(".ppn"):
            keywords       = None
            keyword_paths  = [kw]
        else:
            keywords       = [kw]
            keyword_paths  = None

        porcupine = pvporcupine.create(
            access_key=self._config.porcupine_access_key,
            keywords=keywords,
            keyword_paths=keyword_paths,
        )

        pa     = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,          # always 16000
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,  # always 512
        )

        logger.info("🎙  Waiting for wake word…")

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue

                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                # Convert raw bytes → list[int] that Porcupine expects
                import struct
                pcm_unpacked = list(struct.unpack_from(f"{porcupine.frame_length}h", pcm))

                result = porcupine.process(pcm_unpacked)
                if result >= 0:
                    logger.info("🔔 Wake word detected!")
                    self._wake_event.set()

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()


# ---------------------------------------------------------------------------
# Voice Assistant — top-level orchestrator
# ---------------------------------------------------------------------------

# State constants — clearer than scattered booleans
class State:
    SLEEPING  = "sleeping"   # waiting for wake word
    LISTENING = "listening"  # Whisper is active, collecting speech
    BUSY      = "busy"       # LLM + TTS running


class VoiceAssistant:
    """
    State machine:

        SLEEPING ──(wake word)──► LISTENING ──(silence commit)──► BUSY
            ▲                                                        │
            └────────────────────(TTS done)─────────────────────────┘
    """

    def __init__(self, config: Config) -> None:
        self._cfg     = config
        self._llm     = LLMClient(config)
        self._metrics = Metrics()

        self._stop       = threading.Event()
        self._state_lock = threading.Lock()
        self._state: str = State.SLEEPING

        self._last_llm_time: float = 0.0
        self._last_prompt:   str   = ""

        # Whisper process handle — kept so we can restart it
        self._whisper_proc: Optional[subprocess.Popen] = None

        # ASR text passes through this queue to the aggregator
        self._text_queue: queue.Queue[str] = queue.Queue()

        # Wake-word thread signals via this event
        self._wake_event = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self._llm.warmup()

        # Start Porcupine — it runs throughout the lifetime of the program
        self._wake_detector = WakeWordDetector(self._cfg, self._wake_event)
        self._wake_detector.start()
        # Paused until we explicitly call resume() after TTS
        self._wake_detector.pause()

        # Start the aggregator (decides when to fire the LLM)
        agg_thread = threading.Thread(target=self._agg_loop, daemon=True, name="Aggregator")
        agg_thread.start()

        # Enter the wake-word → listen → respond cycle
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down…")
        finally:
            self._stop.set()
            self._wake_detector.stop()
            self._kill_whisper()
            sys.exit(0)

    # ── Main loop (wake-word gating) ──────────────────────────────────────

    def _main_loop(self) -> None:
        """
        Blocks waiting for a wake event, then starts ASR.
        ASR runs until _run_llm() completes, which re-enables detection.
        """
        self._wake_detector.resume()   # first time: start listening for wake word

        while not self._stop.is_set():
            # Block until wake word fires (or we check every 0.5 s)
            fired = self._wake_event.wait(timeout=0.5)
            if not fired:
                continue

            self._wake_event.clear()

            with self._state_lock:
                if self._state != State.SLEEPING:
                    # Already active — ignore spurious detection
                    continue
                self._state = State.LISTENING

            self._wake_detector.pause()   # stop Porcupine mic reads during ASR

            # Brief discard window so the keyword itself is not transcribed
            logger.info("⏳ Discarding %.1fs post-wake audio…", self._cfg.wake_word_discard_secs)
            time.sleep(self._cfg.wake_word_discard_secs)

            # Start (or restart) whisper-stream
            self._start_whisper()

    def _go_to_sleep(self) -> None:
        """Called after TTS finishes to return to SLEEPING state."""
        self._kill_whisper()
        self._drain_text_queue()

        with self._state_lock:
            self._state = State.SLEEPING

        self._wake_detector.resume()    # re-arm Porcupine

    # ── Whisper management ────────────────────────────────────────────────

    def _start_whisper(self) -> None:
        """Spawn a fresh whisper-stream subprocess and attach a reader thread."""
        self._kill_whisper()   # ensure no stale process

        cmd = [
            self._cfg.whisper_bin,
            "-m",    self._cfg.whisper_model,
            "--step",   str(self._cfg.whisper_step),
            "--length", str(self._cfg.whisper_length),
            "-t",    str(self._cfg.whisper_threads),
            "-ac",   str(self._cfg.whisper_audio_ctx),
        ]
        self._whisper_proc = subprocess.Popen(
            cmd,
            cwd=self._cfg.whisper_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # Reader thread: forwards lines from whisper stdout → _text_queue
        reader = threading.Thread(
            target=self._asr_reader,
            args=(self._whisper_proc,),
            daemon=True,
            name="ASR-reader",
        )
        reader.start()
        logger.info("✅ Whisper started — listening…")

    def _kill_whisper(self) -> None:
        if self._whisper_proc and self._whisper_proc.poll() is None:
            self._whisper_proc.terminate()
            try:
                self._whisper_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._whisper_proc.kill()
        self._whisper_proc = None

    def _asr_reader(self, proc: subprocess.Popen) -> None:
        """Read stdout of a whisper-stream process; forward valid lines to the queue."""
        for line in proc.stdout:  # type: ignore[union-attr]
            if self._stop.is_set():
                break
            with self._state_lock:
                active = self._state == State.LISTENING
            if not active:
                continue
            line = line.strip()
            if not line:
                continue
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                self._text_queue.put(line)

    # ── Aggregator loop ───────────────────────────────────────────────────

    def _agg_loop(self) -> None:
        buffer      = ""
        last_update = time.monotonic()

        while not self._stop.is_set():
            with self._state_lock:
                state = self._state

            # Drain incoming ASR text only while LISTENING
            try:
                text = self._text_queue.get(timeout=0.1)
                if state == State.LISTENING and text.strip():
                    buffer     += " " + text.strip()
                    last_update = time.monotonic()
                    print(f"\r🎤 {buffer}", end="", flush=True)
            except queue.Empty:
                pass

            if state != State.LISTENING:
                buffer      = ""
                last_update = time.monotonic()
                continue

            # Silence-based commit
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

    # ── LLM + TTS pipeline ────────────────────────────────────────────────

    def _run_llm(self, prompt: str, speech_end: float) -> None:
        self._metrics.reset()
        self._metrics.speech_end        = speech_end
        self._metrics.llm_request_start = time.monotonic()

        with self._state_lock:
            self._state       = State.BUSY
            self._last_prompt = prompt

        # Stop Whisper while LLM/TTS is running — saves CPU and avoids
        # Porcupine + Whisper competing for the mic.
        self._kill_whisper()

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

            with self._state_lock:
                self._last_llm_time = time.monotonic()

            self._metrics.log()

            # Return to SLEEPING — re-arms Porcupine
            self._go_to_sleep()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _drain_text_queue(self) -> None:
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    VoiceAssistant(Config()).run()


if __name__ == "__main__":
    main()
