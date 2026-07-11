# Benchmarks

All figures below are transcribed directly from the thesis (Chapter 4, "Results and Evaluation") unless marked
as a **placeholder**. Where the thesis reports mean ± standard deviation, that is preserved. Raw perplexity
scores are **not comparable across different base models** due to differing tokenizers — only compare within
the same model family.

## 1. Perplexity Degradation from Quantization

**Model:** Qwen2.5-0.5B-Instruct, evaluated on WikiText-V2, 512-token context window.

| Precision | Model Size | Perplexity (PPL) | Absolute Increase | % Increase vs FP16 |
|---|---|---|---|---|
| FP16 (baseline) | 1.27 GB | 13.90 ± 0.10 | — | — |
| Q8_0 (8-bit) | 676 MB | 15.76 ± 0.12 | +1.86 | +13.36% |
| Q4_0 (4-bit, naive) | 429 MB | 17.45 ± 0.13 | +3.55 | +25.53% |
| **Q4_K_M (selected)** | **491 MB** | **16.28 ± 0.12** | **+2.38** | **+17.09%** |

**Observation:** Q4_K_M's incremental quality loss vs. Q8_0 is only +0.52 PPL (marginal), while saving ~185 MB
— the thesis's rationale for selecting Q4_K_M as the deployment format over both flat 4-bit and 8-bit
quantization.

## 2. llama.cpp Inference Performance — Raspberry Pi 4B (CPU-only)

**Model:** Qwen2.5-0.5B-Instruct. PP50 = prompt processing speed for 50 tokens. TG50 = token generation
speed for 50 tokens.

| Quantization | Size (MiB) | Threads | PP50 (t/s) | TG50 (t/s) |
|---|---|---|---|---|
| BF16 (unquantized) | 942.43 | 2 | 9.09 ± 0.01 | 5.27 ± 0.01 |
| BF16 | 942.43 | 3 | 9.08 ± 0.07 | 5.36 ± 0.08 |
| BF16 | 942.43 | 4 | 9.05 ± 0.06 | 5.36 ± 0.05 |
| Q8_0 | 638.74 | 2 | 10.13 ± 0.03 | 5.48 ± 0.04 |
| Q8_0 | 638.74 | 3 | 10.11 ± 0.06 | 5.47 ± 0.05 |
| Q8_0 | 638.74 | 4 | 10.05 ± 0.06 | 5.44 ± 0.03 |
| **Q4_K_M** | **462.96** | **2** | **9.00 ± 0.04** | **7.00 ± 0.01** |
| Q4_K_M | 462.96 | 3 | 8.97 ± 0.03 | 7.00 ± 0.02 |
| Q4_K_M | 462.96 | 4 | 8.95 ± 0.04 | 6.94 ± 0.01 |

**Observation:** Adding threads beyond 2 gave flat or slightly worse performance on the Pi 4B — consistent
with `tune_whisper.py`'s own thread sweep (`[2, 3]`) and the deployment default (`--threads 2`).

## 3. Inference Performance — NVIDIA Jetson Orin Nano

**Model:** Gemma3-1B-Instruct, Q4_K_M, average of 5 runs.

| Model | Quantization | Prompt Processing (t/s) | Token Generation (t/s) |
|---|---|---|---|
| Gemma 3 1B | Q4_K_M (4-bit) | 7.77 | 5.27 |

> Thread count was fixed at 2 for this run; the thesis notes GPU (CUDA) inference was used, so CPU thread
> count was deliberately not tuned further.

## 4. End-to-End LLM Comparison (25-token prompt)

| Metric | Raspberry Pi 4B (Qwen2.5-0.5B-Q4_K_M) | Jetson Orin Nano (Gemma3-1B-Q4_K_M) |
|---|---|---|
| Prompt Processing Speed | 9.0 ± 0.45 t/s | 7.77 ± 0.28 t/s |
| Token Generation Speed | 7.0 ± 0.35 t/s | 5.27 ± 0.22 t/s |
| Time to First Token (TTFT) | 5.5 ± 1.3 s | 4.11 ± 0.45 s |
| End-to-End Response Time* | 18 ± 2.8 s | 14 ± 1.4 s |

\* Includes streaming ASR + LLM inference + response assembly + TTS synthesis. Mean ± std. dev. over multiple
runs (exact N not reported in thesis — **placeholder**: record and report sample count in future runs).

**Observation (from thesis):** the Pi 4B shows higher latency than the Jetson due to the absence of hardware
acceleration; nonetheless the lightweight 0.5B model on the Pi remains capable of producing practically usable
speech conversation, while the Jetson's larger 1B model produces qualitatively more natural conversation.

## 5. Memory Profiling — Raspberry Pi 4B (4 GB RAM)

Measured with streaming ASR, LLM inference, and TTS synthesis running simultaneously (steady-state).

| Component | Approximate Memory Usage |
|---|---|
| Local LLM (Qwen2.5-0.5B-Instruct, Q4_K_M) | ~398–650 MB |
| Streaming ASR (whisper.cpp — Medium English) | ~700–950 MB |
| TTS (Piper) | ~250–400 MB |
| System overhead (Python threads, buffers, queues, OS) | ~200–350 MB |
| **Total peak memory usage** | **~1.8–2.2 GB** |

**Observation:** whisper.cpp (Medium model) is the single largest memory consumer, ahead of the LLM. The
thesis notes this leaves enough headroom under the Pi's 4 GB budget for the small (~398 MB quantized) LLM to
coexist comfortably.

> ⚠️ **Note on model mismatch:** Table 4.5 in the thesis references a whisper.cpp *Medium* English model for
> the memory profiling run, whereas §3.3.3 and the LLM inference tables specify `tiny.en`/`base.en` elsewhere
> in the pipeline. Treat the Medium-model memory figure as representative of a heavier ASR configuration, not
> necessarily the exact config used in the latency benchmarks above.

## 6. Jetson Orin Nano Memory Profile

> **Placeholder** — no equivalent memory breakdown table for the Jetson deployment was reported in the thesis.
> Recommended follow-up: profile `jetson_dep.py` under load with `tegrastats` and populate this section.

## 7. Wake Word Detection Accuracy

| Metric | Value |
|---|---|
| Custom "Hey Carat" model validation accuracy | 92% |
| False positive rate | Not reported as non-zero in thesis ("without false positives") — **placeholder**: quantify FPR on a held-out negative set |
| Training data source | Synthetic, generated via Piper TTS with prosody/noise/speaker-identity augmentation |

## 8. Reproducing These Benchmarks

- LLM throughput/perplexity: use `llama.cpp`'s `llama-bench` and `llama-perplexity` tools directly against the
  GGUF files under `models/`.
- Whisper streaming parameter sweep: `python tune_whisper.py --auto` (writes `sweep_results.json` and prints a
  leaderboard + recommended command).
- End-to-end latency: instrumented automatically by `Metrics.log()` in each assistant script — printed after
  every conversational turn.

## 9. Suggested Additions for Future Benchmark Runs

- [ ] Wall-clock wake-word-to-audio-response latency distribution (histogram), not just mean/TTFT.
- [ ] Power draw (W) per device during active inference vs. idle listening.
- [ ] ASR word-error-rate (WER) on a fixed test set, across model sizes (`tiny.en` vs `base.en` vs
      `large-v3-turbo-q5_0`).
- [ ] Jetson memory profile (see §6).
- [ ] Sample size (N runs) explicitly reported alongside every mean ± std. dev. figure.
