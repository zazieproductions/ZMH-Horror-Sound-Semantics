"""
End-to-End Automated Dataset Production Pipeline for ZMH Horror Sound Semantics (v0.1).

Implements the reproducible workflow:
    Generate → Analyze → Propose Labels → Annotate (Label Studio Tasks) → Validate → Export

Supports:
- Batch procedural audio generation with deterministic seeds
- Acoustic feature extraction
- Provisional semantic label & candidate caption generation
- Duplicate detection (by ID, filename, and SHA-256 hash)
- Label Studio task export and human review annotation ingestion
- Full schema, audio, and split-leakage validation
- Idempotent re-execution without duplicating records
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .export_dataset import export_all_dataset_artifacts, load_from_jsonl
from .generate_audio import generate_dataset_audio
from .generate_metadata import generate_dataset_metadata
from .label_studio_integration import ingest_human_annotations_from_label_studio
from .validate_dataset import validate_full_dataset

logger = logging.getLogger(__name__)


def run_pipeline(
    base_dir: Optional[Path] = None,
    count: int = 100,
    sample_rate: int = 16000,
    base_seed: int = 202610,
    human_annotations_export_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute the complete Generate -> Analyze -> Propose Labels -> Annotate -> Validate -> Export
    pipeline idempotently.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    audio_dir = data_dir / "demo_audio"
    jsonl_path = data_dir / "training_data.jsonl"

    # Load existing records if present so human-reviewed records are preserved on re-runs
    existing_records: List[Dict[str, Any]] = []
    if jsonl_path.exists():
        try:
            existing_records = load_from_jsonl(jsonl_path)
            logger.info("Loaded %d existing records for idempotent merge.", len(existing_records))
        except Exception as exc:
            logger.warning("Could not load existing JSONL (%s); regenerating cleanly.", exc)

    # Step 1: Generate procedural audio samples
    specs = generate_dataset_audio(
        output_dir=audio_dir,
        count=count,
        sample_rate=sample_rate,
        base_seed=base_seed,
    )

    # Step 2 & 3: Analyze audio & propose metadata/labels
    records = generate_dataset_metadata(specs=specs, existing_records=existing_records)

    # Step 4: Optional ingestion of completed human review annotations from Label Studio
    review_ingest_stats = None
    if human_annotations_export_path and Path(human_annotations_export_path).exists():
        ls_exported = json.loads(
            Path(human_annotations_export_path).read_text(encoding="utf-8")
        )
        records, review_ingest_stats = ingest_human_annotations_from_label_studio(
            ls_exported_tasks=ls_exported,
            existing_records=records,
        )

    # Step 5: Export JSONL, CSV, Provenance Manifest, and Label Studio Tasks
    exported_paths = export_all_dataset_artifacts(records=records, data_dir=data_dir)

    # Step 6: Validate full dataset
    validation_report = validate_full_dataset(base_dir=base_dir)
    if not validation_report["valid"]:
        for err in validation_report["errors"]:
            logger.error("Validation error: %s", err)
        raise RuntimeError(
            f"Pipeline validation failed with {validation_report['error_count']} errors."
        )

    return {
        "pipeline_status": "success",
        "generated_audio_count": len(specs),
        "exported_paths": exported_paths,
        "review_ingest_stats": review_ingest_stats,
        "validation_report": validation_report,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the ZMH Horror Sound Semantics end-to-end dataset pipeline."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Base directory of horror-sound-semantics",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[16000, 22050, 44100, 48000],
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=202610,
        help="Base random seed for deterministic generation",
    )
    parser.add_argument(
        "--ingest-reviews",
        type=Path,
        default=None,
        help="Optional path to Label Studio completed annotations export JSON",
    )
    args = parser.parse_args()
    summary = run_pipeline(
        base_dir=args.base_dir,
        count=args.count,
        sample_rate=args.sample_rate,
        base_seed=args.base_seed,
        human_annotations_export_path=args.ingest_reviews,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
