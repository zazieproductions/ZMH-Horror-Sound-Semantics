"""Tests for CSV/JSONL exports, sample-to-metadata correspondence, and idempotent pipeline runs."""

from pathlib import Path

from src.pipeline import run_pipeline
from src.validate_dataset import validate_full_dataset

BASE_DIR = Path(__file__).resolve().parent.parent


def test_full_dataset_validation_passes_on_production_data() -> None:
    """Full validation across all 200 records, WAV files, CSVs, JSONL, and XML must pass."""
    report = validate_full_dataset(BASE_DIR)
    assert report["valid"] is True
    assert report["error_count"] == 0
    assert report["total_records"] == 200
    assert report["human_reviewed_count"] == 0
    assert report["review_status_breakdown"] == {"unreviewed_provisional": 200}


def test_idempotent_pipeline_execution_does_not_duplicate_records(tmp_path: Path) -> None:
    """Running the full pipeline twice on the same directory must not duplicate records."""
    # Set up temporary dataset workspace with copies of configs
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "dataset_schema.json").write_text(
        (BASE_DIR / "configs" / "dataset_schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (configs_dir / "annotation_config.xml").write_text(
        (BASE_DIR / "configs" / "annotation_config.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    run1 = run_pipeline(base_dir=tmp_path, count=20, sample_rate=16000, base_seed=202610)
    assert run1["validation_report"]["valid"] is True
    assert run1["validation_report"]["total_records"] == 20

    run2 = run_pipeline(base_dir=tmp_path, count=20, sample_rate=16000, base_seed=202610)
    assert run2["validation_report"]["valid"] is True
    assert run2["validation_report"]["total_records"] == 20
