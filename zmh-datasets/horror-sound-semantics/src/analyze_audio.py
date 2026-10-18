"""
Acoustic Feature Extraction Engine for ZMH Horror Sound Semantics (v0.1).

Extracts deterministic, factual signal-processing metrics from generated or imported
16-bit PCM WAV files without relying on heavy pretrained neural networks.

Note on IP Classification:
- Factual acoustic measurements (RMS energy, spectral centroid, rolloff, ZCR, etc.)
  are objective mathematical measurements and are classified as noncopyrightable factual
  metadata under `docs/IP_OWNERSHIP.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisResult:
    """Factual header and spectral/temporal signal measurements for an audio file."""

    filepath: str
    sample_rate: int
    num_channels: int
    sample_width_bytes: int
    num_frames: int
    duration: float
    acoustic_features: Dict[str, float]


def read_wav_as_float64(filepath: Path) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Read a WAV file and return (mono_float64_signal, sample_rate, nchannels, sampwidth, nframes).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    with wave.open(str(filepath), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)

    if nframes == 0 or len(raw_bytes) == 0:
        raise ValueError(f"Audio file contains zero frames: {filepath}")

    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width ({sampwidth} bytes) in {filepath}")

    if nchannels > 1:
        data = data.reshape(-1, nchannels).mean(axis=1)

    return data, sample_rate, nchannels, sampwidth, nframes


def extract_acoustic_features(signal_f64: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Calculate spectral and temporal acoustic features from a normalized mono waveform.
    """
    n = len(signal_f64)
    if n == 0:
        raise ValueError("Cannot analyze empty waveform array.")

    eps = 1e-12
    rms = float(np.sqrt(np.mean(signal_f64 ** 2)))
    peak = float(np.max(np.abs(signal_f64)))
    crest_db = float(20.0 * math.log10(max(peak, eps) / max(rms, eps)))

    # Zero-crossing rate
    signs = np.signbit(signal_f64)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if n > 1 else 0.0

    # Windowed FFT magnitude/power spectrum across frames (frame size 1024, hop 512)
    frame_len = min(1024, n)
    hop_len = max(1, frame_len // 2)
    window = np.hanning(frame_len)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)

    frame_powers = []
    frame_rms_env = []
    for start in range(0, max(1, n - frame_len + 1), hop_len):
        frame = signal_f64[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        spec = np.fft.rfft(frame * window)
        power = np.abs(spec) ** 2
        frame_powers.append(power)
        frame_rms_env.append(float(np.sqrt(np.mean(frame ** 2))))

    mean_power = np.mean(np.vstack(frame_powers), axis=0) + eps
    total_power = float(np.sum(mean_power))

    # Spectral centroid & bandwidth
    centroid = float(np.sum(freqs * mean_power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_power) / total_power))

    # Spectral rolloff (85%)
    cumulative_power = np.cumsum(mean_power)
    rolloff_idx = int(np.searchsorted(cumulative_power, 0.85 * total_power))
    rolloff_idx = min(len(freqs) - 1, max(0, rolloff_idx))
    rolloff_85 = float(freqs[rolloff_idx])

    # Spectral flatness (Wiener entropy: geometric mean / arithmetic mean)
    geom_mean = float(np.exp(np.mean(np.log(mean_power))))
    arith_mean = float(np.mean(mean_power))
    flatness = float(min(1.0, max(0.0, geom_mean / max(arith_mean, eps))))

    # Low (<250 Hz) and High (>2000 Hz) frequency energy ratios
    low_mask = freqs <= 250.0
    high_mask = freqs >= 2000.0
    low_ratio = float(np.sum(mean_power[low_mask]) / total_power)
    high_ratio = float(np.sum(mean_power[high_mask]) / total_power)

    # Onset strength std (positive envelope derivative variability)
    env_arr = np.array(frame_rms_env, dtype=np.float64)
    if len(env_arr) > 1:
        onset_diff = np.maximum(0.0, np.diff(env_arr))
        onset_std = float(np.std(onset_diff) / max(np.mean(env_arr), eps))
    else:
        onset_std = 0.0

    # Normalized temporal centroid in [0.0, 1.0]
    sq_energy = signal_f64 ** 2 + eps
    time_idx = np.linspace(0.0, 1.0, n, endpoint=False)
    temporal_centroid = float(np.sum(time_idx * sq_energy) / np.sum(sq_energy))

    return {
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "crest_factor_db": round(crest_db, 4),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_bandwidth_hz": round(bandwidth, 2),
        "spectral_rolloff_85_hz": round(rolloff_85, 2),
        "spectral_flatness": round(flatness, 6),
        "low_frequency_energy_ratio": round(low_ratio, 6),
        "high_frequency_energy_ratio": round(high_ratio, 6),
        "onset_strength_std": round(onset_std, 6),
        "temporal_centroid_norm": round(min(1.0, max(0.0, temporal_centroid)), 6),
    }


def analyze_wav_file(filepath: Path) -> AudioAnalysisResult:
    """Analyze a WAV file and return its complete AudioAnalysisResult."""
    data, sr, nchannels, sampwidth, nframes = read_wav_as_float64(filepath)
    duration = round(float(nframes) / float(sr), 4)
    features = extract_acoustic_features(data, sr)
    return AudioAnalysisResult(
        filepath=str(filepath),
        sample_rate=sr,
        num_channels=nchannels,
        sample_width_bytes=sampwidth,
        num_frames=nframes,
        duration=duration,
        acoustic_features=features,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze WAV audio files and extract acoustic features.")
    parser.add_argument("audio_path", type=Path, help="Path to a .wav file or directory of .wav files")
    args = parser.parse_args()

    if args.audio_path.is_dir():
        wav_files = sorted(args.audio_path.glob("*.wav"))
        results = [asdict(analyze_wav_file(p)) for p in wav_files]
        print(json.dumps({"count": len(results), "samples": results[:3]}, indent=2))
    else:
        res = analyze_wav_file(args.audio_path)
        print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    main()
