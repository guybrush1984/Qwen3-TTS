# CLAUDE.md - Qwen3-TTS Developer Guide

## Quick Start

```python
from qwen_tts import Qwen3TTSModel
import torch

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16
)

# Single generation
wavs, sr = tts.generate_custom_voice(text="Hello!", speaker="Vivian", language="English")

# Batch generation (native support)
wavs, sr = tts.generate_custom_voice(
    text=["Text 1", "Text 2", ..., "Text 16"],
    speaker="Vivian",
    language="English"
)
```

---

## What is Qwen3-TTS?

A text-to-speech system using an LLM to generate audio codes, then decoding them to waveforms:

```
Text → LLM (generates audio codes) → Decoder (codes → waveform) → Audio (24kHz)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Qwen3-TTS System                              │
│                                                                         │
│   ┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐  │
│   │   Text      │────▶│   Talker Model      │────▶│  Speech         │  │
│   │   Input     │     │   (32-layer LLM)    │     │  Tokenizer      │──┼──▶ Waveform
│   └─────────────┘     │   + Sub-Talker      │     │  (Decoder)      │  │    (24kHz)
│                       └─────────────────────┘     └─────────────────┘  │
│   ┌─────────────┐     ┌─────────────────────┐                          │
│   │  Reference  │────▶│  Speaker Encoder    │                          │
│   │  Audio      │     │  (ECAPA-TDNN)       │                          │
│   └─────────────┘     └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three Core Components

| Component | Purpose | Key File |
|-----------|---------|----------|
| **Talker Model** | LLM generating audio codes | `qwen_tts/core/models/modeling_qwen3_tts.py` |
| **Speech Tokenizer** | Encode audio→codes, decode codes→audio | `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| **Speaker Encoder** | Extract speaker embedding from reference | Built into main model (ECAPA-TDNN) |

---

## The Three Model Variants

### 1. Base Model - Voice Cloning
```python
# Clone any voice from ~3s reference audio
wavs, sr = tts.generate_voice_clone(
    text="Hello!",
    ref_audio="sample.wav",
    ref_text="Transcript of sample",  # Required for ICL mode
    x_vector_only_mode=False          # False=ICL mode (better), True=embedding only (faster)
)
```
- Uses reference audio at inference time
- Two modes: ICL (uses ref audio + text) or X-Vector only (just speaker embedding)

### 2. CustomVoice Model - Predefined Speakers
```python
# Use predefined speakers with optional instruction
wavs, sr = tts.generate_custom_voice(
    text="Hello!",
    speaker="Vivian",  # Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    language="English",
    instruct="Speak slowly"  # Optional
)
```
- 9 predefined speakers
- No reference audio needed
- Supports instruction-based control

### 3. VoiceDesign Model - Voice from Description
```python
# Create voice from natural language description
wavs, sr = tts.generate_voice_design(
    text="Hello!",
    instruct="A warm, friendly female voice with British accent"
)
```
- No speaker ID or reference audio
- Pure text-based voice design

### Fine-Tuning vs Voice Cloning

| Aspect | Voice Cloning | Fine-Tuning |
|--------|---------------|-------------|
| **Setup** | Instant (3s audio clip) | Training required |
| **Quality** | Good, may have artifacts | Higher fidelity |
| **Runtime** | Process ref audio each call | Speaker baked into weights |
| **Flexibility** | Any voice on the fly | Fixed to trained speakers |

---

## Key Code Paths

### Inference Flow

```
1. generate_custom_voice()          → qwen_tts/inference/qwen3_tts_model.py:732
2. Text tokenization                → Uses Qwen2Tokenizer
3. model.generate()                 → qwen_tts/core/models/modeling_qwen3_tts.py:2022
   ├── Build input embeddings       → Lines 2068-2237
   ├── Left-pad for batching        → Lines 2239-2254
   ├── talker.generate()            → Line 2272 (autoregressive LLM generation)
   └── Extract codes per sample     → Lines 2280-2292
4. speech_tokenizer.decode()        → qwen_tts/inference/qwen3_tts_tokenizer.py:259
   └── model.decode()               → tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:993
       └── decoder.chunked_decode() → Line 1015, processes in chunks of 300 frames
```

### Talker Model (LLM)

Located: `qwen_tts/core/models/modeling_qwen3_tts.py`

- 32 transformer layers, 32 attention heads, 4096 hidden size
- Generates `codec_0` autoregressively
- **Sub-Talker** predicts codebooks 1-15 from hidden states (parallel)
- Output: (batch, seq_len, 16) audio codes

### Speech Tokenizer (12Hz)

Located: `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`

**Encoding** (audio → codes):
- Input: 24kHz waveform
- Mimi encoder → RVQ (16 quantizers, 2048 codebook size)
- Output: (T, 16) codes at 12.5 fps

**Decoding** (codes → audio):
- RVQ decode → Pre-conv → 8-layer transformer → Upsampling (rates: 8,5,4,3)
- `chunked_decode()` at line 886: processes in chunks of 300 frames with 25-frame context
- Output: 24kHz waveform

### Batch Processing

The model natively supports batching:
- Variable-length inputs are **left-padded** (lines 2239-2254)
- Attention masks handle padding
- Returns list of variable-length outputs

---

## Directory Structure

```
Qwen3-TTS/
├── qwen_tts/
│   ├── __init__.py                     # Exports Qwen3TTSModel, Qwen3TTSTokenizer
│   ├── __main__.py                     # CLI: python -m qwen_tts
│   ├── cli/demo.py                     # Gradio web UI
│   │
│   ├── core/
│   │   ├── models/
│   │   │   ├── modeling_qwen3_tts.py         # Main TTS model (2299 lines)
│   │   │   ├── configuration_qwen3_tts.py    # Config classes
│   │   │   └── processing_qwen3_tts.py       # Processor wrapper
│   │   │
│   │   ├── tokenizer_12hz/
│   │   │   ├── modeling_qwen3_tts_tokenizer_v2.py  # Encoder + Decoder (1027 lines)
│   │   │   └── configuration_qwen3_tts_tokenizer_v2.py
│   │   │
│   │   └── tokenizer_25hz/                   # Legacy, not used
│   │
│   └── inference/
│       ├── qwen3_tts_model.py          # Qwen3TTSModel wrapper (878 lines)
│       └── qwen3_tts_tokenizer.py      # Qwen3TTSTokenizer wrapper (411 lines)
│
├── finetuning/
│   ├── prepare_data.py                 # Pre-compute audio codes from dataset
│   ├── sft_12hz.py                     # Training script
│   ├── dataset.py                      # PyTorch Dataset
│   └── README.md
│
├── examples/                           # Usage examples
│   ├── test_model_12hz_base.py
│   ├── test_model_12hz_custom_voice.py
│   └── test_model_12hz_voice_design.py
│
└── pyproject.toml                      # Dependencies
```

---

## Configuration

### Model Config
- `tts_model_type`: "base" | "custom_voice" | "voice_design"
- `tts_model_size`: "1.7b" | "0.6b"
- `tokenizer_type`: "qwen3_tts_tokenizer_12hz"

### Generation Parameters
```python
# Defaults from generate_config.json
do_sample=True, top_k=50, top_p=1.0, temperature=0.9
repetition_penalty=1.05, max_new_tokens=2048
# Sub-talker (codebooks 1-15)
subtalker_dosample=True, subtalker_top_k=50, subtalker_temperature=0.9
```

### Dependencies
```
transformers==4.57.3  # Exact version required
accelerate==1.12.0
torch, torchaudio
librosa, soundfile, sox
einops, gradio
```

---

## Fine-Tuning Workflow

```bash
# 1. Prepare data (pre-compute audio codes)
python finetuning/prepare_data.py \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# 2. Train
python finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl train_with_codes.jsonl \
  --speaker_name my_speaker

# 3. Use
tts = Qwen3TTSModel.from_pretrained("./output")
wavs, sr = tts.generate_custom_voice(text, speaker="my_speaker")
```

---

## Performance Notes

### Observed Behavior (Modal L4, 16 parallel streams)
- GPU utilization: ~20%
- CPU utilization: 100% (1 vCPU maxed)
- Duration: 3-4 minutes for long inference

**Root cause not yet identified.** Profiling needed to determine actual bottlenecks.

### Areas to Profile
- Text tokenization
- LLM generation (GPU)
- Audio decoding (`chunked_decode`)
- Audio resampling (`librosa.resample`)
- GPU↔CPU transfers
- Numpy conversions

---

## Common Commands

```bash
# Run Gradio demo
python -m qwen_tts

# Run examples
python examples/test_model_12hz_custom_voice.py
```
