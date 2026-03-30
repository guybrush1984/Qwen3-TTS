"""Analyze training audio files for clipping, levels, and spectral anomalies.

Splits each file into 1-second windows and reports:
- Peak level (dBFS), RMS level, clipping (samples > 0.99)
- Spectral flatness per window (detect buzz/saturation)
- HF energy ratio (detect super-resolution artifacts)
- Flags windows with clipping or anomalous spectral characteristics

Usage:
    cd /home/marc/voice-store && uv run python3 /path/to/analyze_training_audio.py /home/marc/mls/denoised
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze_windows(data: np.ndarray, sr: int, win_sec: float = 1.0):
    """Analyze audio in 1-second windows."""
    win_len = int(win_sec * sr)
    hop = win_len  # non-overlapping
    results = []

    for start in range(0, len(data) - win_len + 1, hop):
        chunk = data[start : start + win_len]
        peak = np.max(np.abs(chunk))
        rms = np.sqrt(np.mean(chunk**2))

        # Clipping: count samples at or above 0.99
        n_clipped = np.sum(np.abs(chunk) >= 0.99)

        # Spectral analysis with 200ms sub-windows
        sub_win = int(0.2 * sr)
        hanning = np.hanning(sub_win)
        freqs = np.fft.rfftfreq(sub_win, 1 / sr)
        hf_mask = freqs >= 8000
        mid_mask = (freqs >= 500) & (freqs < 4000)

        flat_vals = []
        hf_ratios = []
        for sub_start in range(0, len(chunk) - sub_win, sub_win // 2):
            w = chunk[sub_start : sub_start + sub_win]
            if np.sqrt(np.mean(w**2)) < 0.005:
                continue
            power = np.abs(np.fft.rfft(w * hanning)) ** 2 + 1e-12
            geo = np.exp(np.mean(np.log(power)))
            flat_vals.append(geo / np.mean(power))

            total_power = np.sum(power)
            hf_power = np.sum(power[hf_mask])
            hf_ratios.append(hf_power / total_power if total_power > 0 else 0)

        results.append({
            "time_s": start / sr,
            "peak": peak,
            "peak_dbfs": 20 * np.log10(peak + 1e-12),
            "rms": rms,
            "rms_dbfs": 20 * np.log10(rms + 1e-12),
            "n_clipped": int(n_clipped),
            "flatness": float(np.mean(flat_vals)) if flat_vals else 0,
            "hf_ratio": float(np.mean(hf_ratios)) if hf_ratios else 0,
        })

    return results


def main():
    root = Path(sys.argv[1])
    speakers = sorted(d for d in root.iterdir() if d.is_dir())

    for speaker_dir in speakers:
        speaker = speaker_dir.name
        wav_files = sorted(speaker_dir.glob("*.wav"))
        if not wav_files:
            continue

        print(f"\n{'='*80}")
        print(f"  SPEAKER: {speaker} ({len(wav_files)} files)")
        print(f"{'='*80}")

        speaker_peaks = []
        speaker_clips = []
        speaker_flatness = []
        speaker_hf = []
        flagged_windows = []

        for wav_path in wav_files:
            data, sr = sf.read(str(wav_path))
            if data.ndim > 1:
                data = data.mean(axis=1)

            dur = len(data) / sr
            file_peak = np.max(np.abs(data))
            file_rms = np.sqrt(np.mean(data**2))
            file_peak_db = 20 * np.log10(file_peak + 1e-12)
            file_rms_db = 20 * np.log10(file_rms + 1e-12)
            total_clipped = np.sum(np.abs(data) >= 0.99)

            print(f"\n  {wav_path.name}  ({dur:.0f}s, {sr}Hz)")
            print(f"    Peak: {file_peak_db:+.1f} dBFS  RMS: {file_rms_db:+.1f} dBFS  "
                  f"Clipped samples: {total_clipped}")

            windows = analyze_windows(data, sr)

            # Find anomalous windows
            for w in windows:
                speaker_peaks.append(w["peak_dbfs"])
                speaker_clips.append(w["n_clipped"])
                speaker_flatness.append(w["flatness"])
                speaker_hf.append(w["hf_ratio"])

                flags = []
                if w["n_clipped"] > 10:
                    flags.append(f"CLIP({w['n_clipped']})")
                if w["peak"] > 0.99:
                    flags.append(f"PEAK({w['peak']:.4f})")
                if w["flatness"] > 0.03:
                    flags.append(f"FLAT({w['flatness']:.4f})")
                if w["hf_ratio"] > 0.15:
                    flags.append(f"HF({w['hf_ratio']:.3f})")

                if flags:
                    flagged_windows.append((wav_path.name, w["time_s"], flags))
                    print(f"    ** {w['time_s']:6.0f}s  peak={w['peak_dbfs']:+.1f}  "
                          f"flat={w['flatness']:.4f}  hf={w['hf_ratio']:.3f}  "
                          f"{' '.join(flags)}")

        # Speaker summary
        if speaker_peaks:
            print(f"\n  --- {speaker} summary ---")
            print(f"    Peak dBFS:  min={min(speaker_peaks):+.1f}  "
                  f"max={max(speaker_peaks):+.1f}  "
                  f"mean={np.mean(speaker_peaks):+.1f}")
            print(f"    Flatness:   min={min(speaker_flatness):.4f}  "
                  f"max={max(speaker_flatness):.4f}  "
                  f"mean={np.mean(speaker_flatness):.4f}  "
                  f"std={np.std(speaker_flatness):.4f}")
            print(f"    HF ratio:   min={min(speaker_hf):.3f}  "
                  f"max={max(speaker_hf):.3f}  "
                  f"mean={np.mean(speaker_hf):.3f}  "
                  f"std={np.std(speaker_hf):.3f}")
            print(f"    Clipped windows: {sum(1 for c in speaker_clips if c > 10)} / "
                  f"{len(speaker_clips)}")
            print(f"    Flagged windows: {len(flagged_windows)}")


if __name__ == "__main__":
    main()
