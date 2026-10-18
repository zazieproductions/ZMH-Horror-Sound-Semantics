"""Tests for procedural audio synthesis (`src/generate_audio.py`)."""

from pathlib import Path

import pytest

from src.analyze_audio import read_wav_as_float64
from src.generate_audio import (
    SOUND_EVENT_CATEGORIES,
    generate_dataset_audio,
    generate_single_sample,
)


def test_all_ten_categories_generate_valid_wav(tmp_path: Path) -> None:
    """Every one of the 10 procedural synthesis categories must produce an openable WAV file."""
    for idx, category in enumerate(SOUND_EVENT_CATEGORIES, start=1):
        spec = generate_single_sample(
            sample_index=idx,
            category=category,
            recipe_idx=1,
            variant_idx=1,
            output_dir=tmp_path,
            sample_rate=16000,
            base_seed=202610,
        )
        wav_path = Path(spec.audio_filepath)
        assert wav_path.exists()
        data, sr, nchannels, sampwidth, nframes = read_wav_as_float64(wav_path)
        assert sr == 16000
        assert nchannels == 1
        assert sampwidth == 2
        assert nframes == int(spec.duration * 16000)
        assert len(spec.sha256_checksum) == 64
        assert len(spec.temporal_events) >= 1


def test_deterministic_regeneration_produces_identical_sha256(tmp_path: Path) -> None:
    """Generating the same sample twice with the same seed must produce bit-identical WAV files."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    spec_a = generate_single_sample(
        sample_index=1,
        category="dissonant_drones",
        recipe_idx=2,
        variant_idx=1,
        output_dir=dir_a,
        sample_rate=16000,
        base_seed=99999,
    )
    spec_b = generate_single_sample(
        sample_index=1,
        category="dissonant_drones",
        recipe_idx=2,
        variant_idx=1,
        output_dir=dir_b,
        sample_rate=16000,
        base_seed=99999,
    )
    assert spec_a.sha256_checksum == spec_b.sha256_checksum
    assert Path(spec_a.audio_filepath).read_bytes() == Path(spec_b.audio_filepath).read_bytes()


def test_different_variants_produce_distinct_waveforms(tmp_path: Path) -> None:
    """Different variant indices within the same recipe must produce distinct SHA-256 hashes."""
    spec_v1 = generate_single_sample(
        sample_index=1,
        category="synthetic_creature_sounds",
        recipe_idx=1,
        variant_idx=1,
        output_dir=tmp_path,
    )
    spec_v2 = generate_single_sample(
        sample_index=2,
        category="synthetic_creature_sounds",
        recipe_idx=1,
        variant_idx=2,
        output_dir=tmp_path,
    )
    assert spec_v1.sha256_checksum != spec_v2.sha256_checksum
    assert spec_v1.recipe_id == spec_v2.recipe_id
    assert spec_v1.split == spec_v2.split


def test_batch_generation_minimum_count_validation(tmp_path: Path) -> None:
    """Batch generator should reject counts smaller than 10."""
    with pytest.raises(ValueError, match="at least 10"):
        generate_dataset_audio(output_dir=tmp_path, count=5)
