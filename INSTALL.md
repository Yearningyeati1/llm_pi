# Installation Guide

CARAT has been validated on two targets: **Raspberry Pi 4B** (CPU-only, 4 GB RAM) and **NVIDIA Jetson Orin
Nano** (CUDA-accelerated, 8 GB RAM). General x86_64/ARM64 Ubuntu instructions are also provided for
development/testing on a desktop before deploying to hardware.

> ⚠️ Model file paths in the scripts (e.g. `/home/robot/Desktop/trial/llm_pi/...`,
> `/home/pi/Desktop/code_llm_pi/...`) are **hardcoded from the original research environment**. Update the
> `Config` class at the top of whichever entry-point script you run (`carat_assistant_wake.py`, `jetson_dep.py`,
> `whisper.py`, `wakeword.py`) to match your own paths before running.

---

## 1. Common prerequisites (all platforms)

```bash
sudo apt update
sudo apt install -y build-essential cmake git wget curl \
    portaudio19-dev libasound2-dev libpulse-dev alsa-utils pulseaudio \
    python3 python3-venv python3-pip pkg-config
```

Clone the repository:

```bash
git clone https://github.com/<org>/carat-voice-assistant.git
cd carat-voice-assistant
```

---

## 2. Ubuntu (desktop / dev environment, x86_64)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This is the recommended environment for **building and smoke-testing** the pipeline before flashing/deploying
to an embedded board — whisper.cpp/llama.cpp builds are much faster here.

---

## 3. Raspberry Pi 4B (CPU-only)

Tested on Raspberry Pi OS (64-bit), 4 GB RAM model.

```bash
# System deps
sudo apt update && sudo apt install -y build-essential cmake git \
    portaudio19-dev libasound2-dev alsa-utils libatlas-base-dev

# Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CPU pinning note
The original tuning scripts (`tune_whisper.py`, `whisper.py`) pin whisper.cpp to CPU cores `0,1` via
`taskset -c 0,1`, reserving the remaining cores for `llama.cpp` and Piper. This is a deliberate resource
allocation for the 4-core Pi 4B and can be adjusted in `WHISPER_CMD` / `run_whisper()` if your workload differs.

### Recommended: verify thread counts
Per the thesis's own sweep results (`tune_whisper.py`), thread counts beyond 2 gave flat-to-worse performance
on the Pi 4B for whisper.cpp. Start with `--threads 2` and only increase if profiling shows benefit.

---

## 4. NVIDIA Jetson Orin Nano (CUDA-accelerated)

### 4.1 JetPack / CUDA setup

Use NVIDIA's JetPack SDK (flashed via SDK Manager or `sdkmanager` CLI) — this provides CUDA, cuDNN, and
TensorRT pre-installed and version-matched to the board's L4T release.

```bash
# Confirm CUDA toolchain is present
nvcc --version
nvidia-smi   # or: tegrastats
```

> 📌 **Placeholder**: exact JetPack/L4T/CUDA version used in the original thesis experiments was not recorded.
> Use the latest JetPack release compatible with your Orin Nano carrier board.

### 4.2 Build llama.cpp with CUDA support

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
```

### 4.3 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Jetson deployment script (`jetson_dep.py`) uses a simpler always-listening pipeline (no wake word,
silence-timeout based turn detection) — see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the difference vs. the
wake-word variant.

---

## 5. Building whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp ../whisper.cpp
cd ../whisper.cpp
cmake -B build
cmake --build build --config Release -j"$(nproc)"
```

This produces `./build/bin/whisper-stream`, used by `whisper.py` / `jetson_dep.py` / `wakeword.py`.

### 5.1 PCM-stdin variant (used by `carat_assistant_wake.py`)

`carat_assistant_wake.py` expects a **custom-built** `whisper-stream-pcm` binary (reads raw PCM via stdin
instead of capturing from a device directly, and emits `### Transcription START/END` markers). This is **not**
part of stock whisper.cpp.

> ⚠️ **Placeholder**: the patch/fork producing `whisper-stream-pcm` is referenced by the code
> (`whisper_cwd = "../whisper.cpp.pcm"`) but its source is not included in this repository. If you don't have
> this fork, use `whisper.py` / `jetson_dep.py` (stock `whisper-stream`) instead, or implement stdin-PCM
> support yourself against upstream `whisper.cpp`'s `stream` example.

### 5.2 Download Whisper models

```bash
cd ../whisper.cpp
bash ./models/download-ggml-model.sh tiny.en     # Pi (fastest)
bash ./models/download-ggml-model.sh base.en     # Pi/Jetson (balanced)
bash ./models/download-ggml-model.sh large-v3-turbo-q5_0   # highest accuracy variant used in carat_assistant_wake.py
```

---

## 6. Building llama.cpp (CPU — Raspberry Pi)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j"$(nproc)"
```

### 6.1 Download / convert GGUF models

```bash
# Example using huggingface-cli — adjust repo IDs to the actual quantized GGUF you intend to use
pip install huggingface_hub
huggingface-cli download <PLACEHOLDER: Qwen2.5-0.5B-Instruct-GGUF-repo> \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/

huggingface-cli download <PLACEHOLDER: Gemma-3-1B-Instruct-GGUF-repo> \
    gemma-3-1b-it-q4_k_m.gguf --local-dir models/
```

See [`MODELS.md`](MODELS.md) for the full model matrix and quantization guidance.

### 6.2 Start the LLM server

```bash
./llama.cpp/build/bin/llama-server \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --port 8080 -c 2048 --threads 2
```

This must be running (and reachable at `LLAMA_URL`, default `http://localhost:8080/v1/chat/completions`)
before starting any of the assistant scripts.

---

## 7. Installing Piper TTS

```bash
mkdir -p piper && cd piper
wget https://github.com/rhasspy/piper/releases/latest/download/piper_<PLATFORM>.tar.gz
tar -xzf piper_<PLATFORM>.tar.gz
```

Replace `<PLATFORM>` with the correct release asset for your architecture (`linux_aarch64` for Pi/Jetson,
`linux_x86_64` for desktop).

### 7.1 Download voices

```bash
# English (used across all scripts)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json

# Hindi (used by carat_assistant_wake.py -l hi)
wget <PLACEHOLDER: hi_IN-pratham-medium.onnx download URL>
wget <PLACEHOLDER: hi_IN-pratham-medium.onnx.json download URL>
```

> The thesis also mentions a **finetuned Piper voice for an Indian-English accent** (§3.1). This finetuned
> checkpoint is not published in this repository — treat `en_US-ryan-low.onnx` as the reproducible baseline
> voice used across the code.

Update `Config.piper_bin` / `Config.piper_models` in the script you're running to point at your installed
binary and voice files.

---

## 8. Python dependencies

```bash
pip install -r requirements.txt
```

See [`requirements.txt`](#requirementstxt) below. Key packages: `pyaudio`, `numpy`, `requests`,
`openwakeword` (wake-word variant only), `pvporcupine` (legacy Porcupine variant only).

---

## 9. Wake word model (openWakeWord variant)

`carat_assistant_wake.py` expects a custom ONNX wake-word model at the path configured in
`Config.wakeword_model_paths` (default in source: `hey_khaa_rat.onnx`, representing "Hey Carat").

- To train your own: follow the [openWakeWord training documentation](https://github.com/dscripka/openWakeWord),
  using Piper-synthesized training data with prosody/noise/speaker augmentation (as described in thesis §3.3.1).
- A pretrained "Hey Carat" model is **not distributed** with this repository — you must train your own wake word
  or substitute one of openWakeWord's stock keywords for testing.

### Legacy: Porcupine variant (`wakeword.py`)

Requires a Picovoice account and access key:

```bash
pip install pvporcupine
```


---

## 10. Running the LLM warm-up + assistant

```bash
# Terminal 1: LLM server
./llama.cpp/build/bin/llama-server -m models/<your-model>.gguf --port 8080 -c 2048

# Terminal 2: assistant
python carat_assistant_wake.py --language en
```

---

## Troubleshooting

### ALSA warnings flooding the console
`carat_assistant_wake.py` includes a `ctypes`-based ALSA error handler suppression at import time. If you're
running a different script and see ALSA spam, either:
- redirect stderr: `python whisper.py 2>/dev/null`, or
- copy the ALSA suppression block from `carat_assistant_wake.py` into your entry script.

### `OSError: [Errno -9996] Invalid input device`
Your `Config.mic_device` index doesn't match your system. List available devices:

```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i)["name"])
```

Update `mic_device` accordingly.

### `whisper-stream` / `whisper-stream-pcm` binary not found
Verify `whisper_cwd` and `whisper_bin` paths in `Config` match your actual whisper.cpp build location, and
that `cmake --build build` completed without errors.

### `Connection refused` to `localhost:8080`
The `llama-server` process isn't running, crashed, or is bound to a different port. Check its terminal output;
confirm `LLAMA_URL` in the script matches the server's actual `--port`.

### High CPU usage / thermal throttling on Raspberry Pi
- Use `tiny.en` or `base.en` Whisper models, not `large-v3-turbo`, on the Pi.
- Add a heatsink/fan — sustained inference on a passively cooled Pi 4B will throttle.
- Run `tune_whisper.py --auto` to find a lower-CPU parameter configuration for your specific board/mic.

### `taskset: failed to set pid's affinity` 
`taskset -c 0,1` requires the process to actually have access to those cores (check `nproc` / `/proc/cpuinfo`);
on non-4-core boards, adjust the `-c` core list in `WHISPER_CMD`.

### Piper produces no audio / `aplay`/`paplay` errors
Confirm your output device with `aplay -l` / `pactl list sinks short`, and that the sample rate passed to
`aplay`/`paplay` (`-r 16000` or `--rate 22050` depending on voice) matches the voice's native sample rate
(check the voice's `.onnx.json` config for `sample_rate`).
