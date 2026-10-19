"""Tests for acoustic feature extraction and physical WAV file verification (`src/analyze_audio.py`)."""

from pathlib import Path

import numpy as np

from src.analyze_audio import analyze_wav_file, read_wav_as_float64

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "demo_audio"


def test_all_demo_audio_files_open_and_analyze_cleanly() -> None:
    """All generated WAV files in data/demo_audio must open and return finite acoustic features."""
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    assert len(wav_files) == 200

    for wav_path in wav_files:
        data, sr, nchannels, sampwidth, nframes = read_wav_as_float64(wav_path)
        assert sr == 16000
        assert nchannels == 1
        assert sampwidth == 2
        assert nframes >= 32000  # >= 2.0 seconds
        assert np.all(np.isfinite(data))

    # Full spectral/temporal feature analysis on a representative subset across categories
    for idx in range(0, len(wav_files), max(1, len(wav_files) // 10)):
        res = analyze_wav_file(wav_files[idx])
        feats = res.acoustic_features
        assert 0.01 < feats["rms_energy"] <= 1.0
        assert 0.5 < feats["peak_amplitude"] <= 1.001
        assert 0.0 <= feats["zero_crossing_rate"] <= 1.0
        assert 20.0 < feats["spectral_centroid_hz"] < 8000.0
        assert 0.0 <= feats["spectral_flatness"] <= 1.001
        assert 0.0 <= feats["low_frequency_energy_ratio"] <= 1.001
        assert 0.0 <= feats["high_frequency_energy_ratio"] <= 1.001
        assert 0.0 <= feats["temporal_centroid_norm"] <= 1.0
