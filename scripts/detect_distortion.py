"""Detect buzzy distortion in generated TTS audio.

Uses spectral flatness (Wiener entropy) in multiple frequency bands, analyzed
in 200ms sub-windows within 1-second segments.  Three complementary rules
catch different distortion types:

1. Full-band flatness mean > threshold: pervasive codec distortion from
   embedding drift.  Higher = more noise-like.
2. 500-1kHz band flatness mean < threshold: speaker-specific buzz where
   energy concentrates at harmonics unnaturally.  Lower = more buzzy.
3. 8-12kHz flatness std > threshold: intermittent high-frequency artifacts
   that come and go, increasing frame-to-frame variance.

A file is flagged if ANY rule triggers.

Calibrated against 48 human-labeled samples across 4 speakers.

Usage:
    python scripts/detect_distortion.py audio1.wav audio2.wav ...
    python scripts/detect_distortion.py --json output_dir/*.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SegmentResult:
    time_s: float
    flat_full: float
    flat_band: float


@dataclass
class DistortionResult:
    path: str
    duration_s: float
    n_segments: int
    flatness_mean: float
    flatness_band_mean: float
    hf_flatness_std: float
    segments: list[SegmentResult] = field(default_factory=list)
    flagged: bool = False
    flag_reason: str = ""


def analyze_file(
    path: str,
    flatness_threshold: float = 0.022,
    band_threshold: float = 0.115,
    hf_std_threshold: float = 0.15,
) -> DistortionResult:
    """Analyze a WAV file for buzzy distortion.

    Computes spectral flatness in 1-second segments (using 200ms sub-windows).
    Three detection rules:
    - Full-band flatness mean > flatness_threshold: pervasive distortion
    - 500-1kHz band flatness mean < band_threshold: speaker-specific buzz
    - 8-12kHz flatness std > hf_std_threshold: intermittent HF artifacts
    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    sub_win = int(0.200 * sr)
    sub_hop = sub_win // 2
    hanning = np.hanning(sub_win)
    freqs = np.fft.rfftfreq(sub_win, 1 / sr)
    band_mask = (freqs >= 500) & (freqs < 1000)
    hf_mask = (freqs >= 8000) & (freqs < 12000)

    seg_dur = int(1.0 * sr)
    seg_hop = int(0.5 * sr)

    segments = []
    all_flat_full = []
    all_flat_band = []
    all_flat_hf = []

    for seg_start in range(0, len(data) - seg_dur, seg_hop):
        seg = data[seg_start : seg_start + seg_dur]
        flat_full_vals = []
        flat_band_vals = []

        for start in range(0, len(seg) - sub_win, sub_hop):
            window = seg[start : start + sub_win]
            rms = np.sqrt(np.mean(window**2))
            if rms < 0.01:
                continue

            power = np.abs(np.fft.rfft(window * hanning)) ** 2 + 1e-12

            geo = np.exp(np.mean(np.log(power)))
            flat_full_vals.append(geo / np.mean(power))

            bp = power[band_mask]
            if len(bp) > 2:
                geo_b = np.exp(np.mean(np.log(bp)))
                flat_band_vals.append(geo_b / np.mean(bp))

            hp = power[hf_mask]
            if len(hp) > 2:
                geo_h = np.exp(np.mean(np.log(hp)))
                all_flat_hf.append(geo_h / np.mean(hp))

        if flat_full_vals:
            ff = float(np.mean(flat_full_vals))
            fb = float(np.mean(flat_band_vals)) if flat_band_vals else 0.0
            segments.append(SegmentResult(
                time_s=round(seg_start / sr, 1), flat_full=ff, flat_band=fb,
            ))
            all_flat_full.extend(flat_full_vals)
            all_flat_band.extend(flat_band_vals)

    if not segments:
        return DistortionResult(
            path=path, duration_s=len(data) / sr,
            n_segments=0, flatness_mean=0.0, flatness_band_mean=0.0,
            hf_flatness_std=0.0,
        )

    mean_full = float(np.mean(all_flat_full))
    mean_band = float(np.mean(all_flat_band)) if all_flat_band else 0.0
    hf_std = float(np.std(all_flat_hf)) if all_flat_hf else 0.0

    reasons = []
    if mean_full > flatness_threshold:
        reasons.append(f"full={mean_full:.4f}>{flatness_threshold}")
    if mean_band < band_threshold:
        reasons.append(f"band={mean_band:.4f}<{band_threshold}")
    if hf_std > hf_std_threshold:
        reasons.append(f"hf_std={hf_std:.4f}>{hf_std_threshold}")

    return DistortionResult(
        path=path,
        duration_s=len(data) / sr,
        n_segments=len(segments),
        flatness_mean=mean_full,
        flatness_band_mean=mean_band,
        hf_flatness_std=hf_std,
        segments=segments,
        flagged=bool(reasons),
        flag_reason=", ".join(reasons),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="WAV files to analyze")
    parser.add_argument("--flatness-threshold", type=float, default=0.022,
                        help="Full-band flatness above which file is flagged (default: 0.022)")
    parser.add_argument("--band-threshold", type=float, default=0.115,
                        help="500-1kHz flatness below which file is flagged (default: 0.115)")
    parser.add_argument("--hf-std-threshold", type=float, default=0.15,
                        help="8-12kHz flatness std above which file is flagged (default: 0.15)")
    parser.add_argument("--segments", action="store_true",
                        help="Show per-segment detail")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = []
    for f in args.files:
        r = analyze_file(
            f, flatness_threshold=args.flatness_threshold,
            band_threshold=args.band_threshold,
            hf_std_threshold=args.hf_std_threshold,
        )
        results.append(r)

    if args.json:
        out = []
        for r in results:
            entry = {
                "path": r.path, "duration_s": round(r.duration_s, 1),
                "n_segments": r.n_segments,
                "flatness_mean": round(r.flatness_mean, 4),
                "flatness_band_mean": round(r.flatness_band_mean, 4),
                "hf_flatness_std": round(r.hf_flatness_std, 4),
                "flagged": r.flagged,
                "flag_reason": r.flag_reason,
            }
            if args.segments:
                entry["segments"] = [
                    {"time_s": s.time_s, "flat_full": round(s.flat_full, 4),
                     "flat_band": round(s.flat_band, 4)}
                    for s in r.segments
                ]
            out.append(entry)
        json.dump(out, sys.stdout, indent=2)
        print()
    else:
        n_flagged = sum(1 for r in results if r.flagged)
        for r in results:
            flag = "FLAGGED" if r.flagged else "ok"
            name = Path(r.path).name
            reason = f"  ({r.flag_reason})" if r.flag_reason else ""
            print(f"  {flag:<8} {name:<40} full={r.flatness_mean:.4f}  "
                  f"band={r.flatness_band_mean:.4f}  hf_std={r.hf_flatness_std:.4f}{reason}")
            if args.segments:
                for s in r.segments:
                    print(f"           {s.time_s:>5.1f}s  full={s.flat_full:.4f}  "
                          f"band={s.flat_band:.4f}")
        print(f"\n  {n_flagged}/{len(results)} flagged")
        sys.exit(1 if n_flagged > 0 else 0)


if __name__ == "__main__":
    main()
