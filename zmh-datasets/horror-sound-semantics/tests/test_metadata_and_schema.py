"""Tests for schema compliance, duplicate detection, provenance rules, and split separation."""

import copy
from pathlib import Path

from src.export_dataset import load_from_jsonl
from src.validate_dataset import (
    validate_provenance_and_semantics,
    validate_records_against_schema,
    validate_splits_and_leakage,
    validate_uniqueness_and_duplicates,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "configs" / "dataset_schema.json"
JSONL_PATH = BASE_DIR / "data" / "training_data.jsonl"


def test_all_records_comply_with_dataset_json_schema() -> None:
    """Every record in training_data.jsonl must pass JSON Schema validation with 0 errors."""
    records = load_from_jsonl(JSONL_PATH)
    assert len(records) == 200
    errors = validate_records_against_schema(records, SCHEMA_PATH)
    assert errors == []


def test_schema_rejects_invalid_emotion_and_missing_fields() -> None:
    """JSON Schema validator must catch invalid controlled vocabulary values."""
    records = load_from_jsonl(JSONL_PATH)
    bad_record = copy.deepcopy(records[0])
    bad_record["primary_emotion"] = "happiness_invalid"
    errors = validate_records_against_schema([bad_record], SCHEMA_PATH)
    assert len(errors) >= 1
    assert "primary_emotion" in errors[0]


def test_uniqueness_and_duplicate_detection() -> None:
    """Validator must confirm 0 duplicates in dataset and detect injected duplicates."""
    records = load_from_jsonl(JSONL_PATH)
    assert validate_uniqueness_and_duplicates(records) == []

    # Inject duplicate sample_id and duplicate sha256_checksum
    dup_records = copy.deepcopy(records[:5])
    dup_records[1]["sample_id"] = dup_records[0]["sample_id"]
    dup_records[2]["sha256_checksum"] = dup_records[0]["sha256_checksum"]
    errors = validate_uniqueness_and_duplicates(dup_records)
    assert any("Duplicate sample_id" in e for e in errors)
    assert any("Duplicate audio SHA-256" in e for e in errors)


def test_provenance_and_ip_boundary_rules() -> None:
    """Procedural audio must not claim authentic field recording or unverified human expert status."""
    records = load_from_jsonl(JSONL_PATH)
    assert validate_provenance_and_semantics(records) == []

    # Inject forbidden claims
    bad = copy.deepcopy(records[0])
    bad["is_authentic_human_or_field_recording"] = True
    bad["annotation_provenance"] = "human_expert_reviewed_v0.1"
    errors = validate_provenance_and_semantics([bad])
    assert len(errors) == 2


def test_stratified_splits_and_zero_recipe_leakage() -> None:
    """Splits must be 120/40/40 stratified and have 0 overlapping recipe_ids across splits."""
    records = load_from_jsonl(JSONL_PATH)
    errors, summary = validate_splits_and_leakage(records)
    assert errors == []
    assert summary["counts_by_split"] == {"train": 120, "validation": 40, "test": 40}
    assert summary["zero_recipe_leakage"] is True

    # Inject cross-split recipe leakage and verify detection
    leaked = copy.deepcopy(records)
    leaked[0]["split"] = "test"  # Variant 1 of dissonant_drones_r01 moved to test while Variant 2 is in train
    leak_errors, leak_summary = validate_splits_and_leakage(leaked)
    assert leak_summary["zero_recipe_leakage"] is False
    assert any("Recipe leakage" in e for e in leak_errors)
