#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "transformers==4.57.3",
#     "soundfile",
#     "numpy",
#     "einops",
#     "librosa",
#     "qwen-tts",
# ]
#
# [tool.uv.sources]
# qwen-tts = { path = ".." }
# ///
"""Round-trip audio through the speech tokenizer codec (encode -> decode).

Tests whether the codec itself introduces distortion by encoding training
WAV files to codes and decoding them back.

Usage:
    uv run scripts/roundtrip_codec.py input_dir/ --output-dir output_dir/ [--n 5]
    uv run scripts/roundtrip_codec.py file1.wav file2.wav --output-dir output_dir/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="WAV files or directories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for round-tripped files")
    parser.add_argument("--n", type=int, default=5,
                        help="Max files per directory (default: 5)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        help="Tokenizer model path")
    args = parser.parse_args()

    # Collect input files
    wav_files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            files = sorted(p.glob("*.wav"))[:args.n]
            wav_files.extend(files)
        elif p.is_file():
            wav_files.append(p)

    if not wav_files:
        print("No WAV files found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.tokenizer}...")
    from qwen_tts import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer, device_map="cpu",
    )

    print(f"Processing {len(wav_files)} files...")
    for wav_path in wav_files:
        orig_name = f"{wav_path.stem}_original.wav"
        rt_name = f"{wav_path.stem}_roundtrip.wav"

        orig_data, orig_sr = sf.read(str(wav_path))

        enc_result = tokenizer.encode(str(wav_path))
        wavs, sr = tokenizer.decode(enc_result)
        rt_data = wavs[0]

        sf.write(os.path.join(args.output_dir, orig_name), orig_data, orig_sr)
        sf.write(os.path.join(args.output_dir, rt_name), rt_data, sr)

        orig_dur = len(orig_data) / orig_sr
        rt_dur = len(rt_data) / sr
        print(f"  {wav_path.name:<50} {orig_dur:.1f}s -> {rt_dur:.1f}s")

    print(f"\nDone. Files in {args.output_dir}/")
    print("Compare *_original.wav vs *_roundtrip.wav")


if __name__ == "__main__":
    main()
