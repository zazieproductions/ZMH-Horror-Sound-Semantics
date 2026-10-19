"""Offline tests for Hub discovery and cautious import behavior."""
import json
from unittest.mock import Mock

import pytest

from src import huggingface_integration as hf


def test_license_screening_is_conservative():
    assert hf.verify_license_metadata({}, "cc0-1.0")["status"] == "manual_review_required"
    restricted = {"cardData": {"description": "Research only, non-commercial dataset"}}
    assert any("restrictions" in note for note in hf.verify_license_metadata(restricted, "cc-by-4.0")["notes"])


def test_preview_is_capped_to_five(monkeypatch):
    calls = []
    monkeypatch.setattr(hf, "_get_json", lambda url, params=None: calls.append(params) or {"rows": [{"row": {"n": i}} for i in range(8)]})
    assert len(hf.preview_dataset("someone/audio", "train")) == 5
    assert calls[0]["length"] == 5
    with pytest.raises(ValueError):
        hf.preview_dataset("someone/audio", count=6)


def test_import_requires_explicit_approval_and_allowlisted_license(monkeypatch, tmp_path):
    with pytest.raises(PermissionError):
        hf.import_approved_records("org/data", output_dir=tmp_path)
    monkeypatch.setattr(hf, "_get_json", lambda url, params=None: {"tags": ["license:cc-by-4.0"], "sha": "abc123", "cardData": {"license": "cc-by-4.0"}} if "/api/datasets/" in url else {"rows": [{"row_idx": 0, "row": {"audio": {"path": "clip.wav"}, "label": "rain"}}]})
    result = hf.import_approved_records("org/data", approve=True, output_dir=tmp_path)
    assert result[0]["source_record_id"] == "org/data:0"
    assert result[0]["source_revision"] == "abc123"
    assert result[0]["attribution_required"] is True
    assert result[0]["human_review_status"] == "unreviewed_provisional"
    assert (tmp_path / "org_data.jsonl").exists()
    # Re-import must be idempotent.
    assert len(hf.import_approved_records("org/data", approve=True, output_dir=tmp_path)) == 1


def test_restricted_license_is_never_imported(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "_get_json", lambda url, params=None: {"tags": ["license:other"]})
    with pytest.raises(PermissionError):
        hf.import_approved_records("org/data", approve=True, output_dir=tmp_path)
