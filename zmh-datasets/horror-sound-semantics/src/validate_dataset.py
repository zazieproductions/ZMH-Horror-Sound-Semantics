"""
Comprehensive Dataset & Audio Validation Suite for ZMH Horror Sound Semantics (v0.1).

Performs:
1. JSON Schema Draft 2020-12 validation of all records against `configs/dataset_schema.json`.
2. Duplicate detection across `sample_id`, `audio_filename`, `sha256_checksum`, and `(recipe_id, variant_index)`.
3. Physical WAV audio verification (openable, valid PCM header, duration/sample_rate match,
   non-silent signal, and exact SHA-256 checksum verification).
4. 1-to-1 correspondence checks across `demo_audio/*.wav`, `training_data.jsonl`,
   `metadata.csv`, `provenance_manifest.csv`, and `label_studio_tasks.json`.
5. Split stratification and zero-leakage verification (`recipe_id` disjointness across
   `train`, `validation`, and `test` splits).
6. Provenance and IP boundary integrity verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema
import numpy as np

from .analyze_audio import read_wav_as_float64
from .export_dataset import (
    METADATA_CSV_COLUMNS,
    PROVENANCE_MANIFEST_COLUMNS,
    load_from_jsonl,
)
from .generate_audio import SOUND_EVENT_CATEGORIES, compute_file_sha256
from .label_studio_integration import validate_annotation_config_xml

logger = logging.getLogger(__name__)


def validate_records_against_schema(
    records: List[Dict[str, Any]],
    schema_path: Path,
) -> List[str]:
    """Validate every record against `configs/dataset_schema.json` and return any errors."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    errors: List[str] = []
    for idx, rec in enumerate(records):
        sid = rec.get("sample_id", f"index_{idx}")
        for err in validator.iter_errors(rec):
            loc = ".".join(str(p) for p in err.path) or "<root>"
            errors.append(f"[{sid}] Schema error at '{loc}': {err.message}")
    return errors


def validate_uniqueness_and_duplicates(records: List[Dict[str, Any]]) -> List[str]:
    """Check uniqueness of sample_id, audio_filename, sha256_checksum, and (recipe_id, variant_index)."""
    errors: List[str] = []
    seen_ids: Set[str] = set()
    seen_files: Set[str] = set()
    seen_hashes: Dict[str, str] = {}
    seen_recipe_variants: Set[Tuple[str, int]] = set()

    for rec in records:
        sid = rec["sample_id"]
        fname = rec["audio_filename"]
        sha = rec["sha256_checksum"]
        rv = (rec["recipe_id"], int(rec["variant_index"]))

        if sid in seen_ids:
            errors.append(f"Duplicate sample_id detected: {sid}")
        seen_ids.add(sid)

        if fname in seen_files:
            errors.append(f"Duplicate audio_filename detected: {fname}")
        seen_files.add(fname)

        if sha in seen_hashes:
            errors.append(
                f"Duplicate audio SHA-256 hash detected between {sid} and {seen_hashes[sha]}: {sha}"
            )
        seen_hashes[sha] = sid

        if rv in seen_recipe_variants:
            errors.append(f"Duplicate (recipe_id, variant_index) detected: {rv}")
        seen_recipe_variants.add(rv)

    return errors


def validate_audio_files(
    records: List[Dict[str, Any]],
    audio_dir: Path,
) -> List[str]:
    """
    Verify every record's WAV file exists, opens cleanly, has matching sample_rate/duration,
    is non-silent and free of NaNs, and matches its recorded SHA-256 checksum.
    """
    errors: List[str] = []
    audio_dir = Path(audio_dir)

    expected_filenames = {r["audio_filename"] for r in records}
    actual_filenames = {p.name for p in audio_dir.glob("*.wav")}

    missing_on_disk = expected_filenames - actual_filenames
    if missing_on_disk:
        errors.append(f"Missing WAV files on disk: {sorted(missing_on_disk)}")

    orphan_on_disk = actual_filenames - expected_filenames
    if orphan_on_disk:
        errors.append(f"Orphan WAV files on disk not in metadata: {sorted(orphan_on_disk)}")

    for rec in records:
        sid = rec["sample_id"]
        wav_path = audio_dir / rec["audio_filename"]
        if not wav_path.exists():
            continue

        try:
            data, sr, nchannels, sampwidth, nframes = read_wav_as_float64(wav_path)
        except Exception as exc:
            errors.append(f"[{sid}] Failed to open WAV file {wav_path.name}: {exc}")
            continue

        if sr != rec["sample_rate"]:
            errors.append(
                f"[{sid}] Sample rate mismatch: WAV={sr} vs metadata={rec['sample_rate']}"
            )

        actual_duration = round(float(nframes) / float(sr), 4)
        if abs(actual_duration - float(rec["duration"])) > 0.01:
            errors.append(
                f"[{sid}] Duration mismatch: WAV={actual_duration}s vs metadata={rec['duration']}s"
            )

        if not np.all(np.isfinite(data)):
            errors.append(f"[{sid}] WAV waveform contains NaN or Inf values.")

        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms < 0.005:
            errors.append(f"[{sid}] WAV waveform is near-silent (RMS={rms:.6f}).")

        disk_sha = compute_file_sha256(wav_path)
        if disk_sha != rec["sha256_checksum"]:
            errors.append(
                f"[{sid}] SHA-256 mismatch: disk={disk_sha} vs metadata={rec['sha256_checksum']}"
            )

    return errors


def validate_provenance_and_semantics(records: List[Dict[str, Any]]) -> List[str]:
    """
    Validate semantic consistency, temporal event bounds, and IP/provenance rules.
    """
    errors: List[str] = []
    for rec in records:
        sid = rec["sample_id"]

        # Primary emotion must not be repeated in secondary_emotions
        if rec["primary_emotion"] in rec["secondary_emotions"]:
            errors.append(
                f"[{sid}] primary_emotion '{rec['primary_emotion']}' is duplicated in secondary_emotions"
            )

        # Procedural audio must not claim to be an authentic human or field recording
        if (
            rec["generation_method"] == "procedural_dsp_synthesis"
            and rec["is_authentic_human_or_field_recording"] is True
        ):
            errors.append(
                f"[{sid}] Procedural audio cannot be marked is_authentic_human_or_field_recording=True"
            )

        # Unreviewed provisional records must not claim human expert provenance
        if rec["human_review_status"] == "unreviewed_provisional" and rec[
            "annotation_provenance"
        ] in ("human_expert_reviewed_v0.1", "human_expert_corrected_v0.1"):
            errors.append(
                f"[{sid}] Record is unreviewed_provisional but claims {rec['annotation_provenance']}"
            )

        # Temporal events must lie within [0.0, duration]
        dur = float(rec["duration"])
        for ev in rec.get("temporal_events", []):
            s_sec = float(ev["start_sec"])
            e_sec = float(ev["end_sec"])
            if s_sec < 0.0 or e_sec > dur + 0.001 or e_sec <= s_sec:
                errors.append(
                    f"[{sid}] Invalid temporal_event bounds [{s_sec}, {e_sec}] for duration {dur}s"
                )

    return errors


def validate_splits_and_leakage(records: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Verify stratified split distribution and confirm zero recipe_id leakage across splits.
    """
    errors: List[str] = []
    recipes_by_split: Dict[str, Set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    counts_by_split: Counter = Counter()
    cat_by_split: Dict[str, Counter] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    for rec in records:
        sp = rec["split"]
        counts_by_split[sp] += 1
        recipes_by_split[sp].add(rec["recipe_id"])
        cat_by_split[sp][rec["sound_event_category"]] += 1

    for sp in ("train", "validation", "test"):
        if counts_by_split[sp] == 0:
            errors.append(f"Split '{sp}' is empty.")

    # Check disjointness of recipe_id across splits
    tr_val = recipes_by_split["train"] & recipes_by_split["validation"]
    tr_te = recipes_by_split["train"] & recipes_by_split["test"]
    val_te = recipes_by_split["validation"] & recipes_by_split["test"]

    if tr_val:
        errors.append(f"Recipe leakage between train and validation splits: {sorted(tr_val)}")
    if tr_te:
        errors.append(f"Recipe leakage between train and test splits: {sorted(tr_te)}")
    if val_te:
        errors.append(f"Recipe leakage between validation and test splits: {sorted(val_te)}")

    # Check category stratification if dataset has >= 50 samples
    if len(records) >= 50:
        for sp in ("train", "validation", "test"):
            missing_cats = set(SOUND_EVENT_CATEGORIES) - set(cat_by_split[sp].keys())
            if missing_cats:
                errors.append(
                    f"Split '{sp}' is missing sound_event_categories: {sorted(missing_cats)}"
                )

    split_summary = {
        "counts_by_split": dict(counts_by_split),
        "unique_recipes_by_split": {k: len(v) for k, v in recipes_by_split.items()},
        "zero_recipe_leakage": not (tr_val or tr_te or val_te),
    }
    return errors, split_summary


def validate_exported_files_correspondence(
    records: List[Dict[str, Any]],
    data_dir: Path,
) -> List[str]:
    """
    Confirm 1-to-1 correspondence across JSONL, metadata.csv, provenance_manifest.csv,
    and label_studio_tasks.json.
    """
    errors: List[str] = []
    data_dir = Path(data_dir)
    expected_ids = [r["sample_id"] for r in sorted(records, key=lambda x: x["sample_id"])]

    # 1. Check metadata.csv
    meta_csv = data_dir / "metadata.csv"
    if not meta_csv.exists():
        errors.append(f"Missing metadata.csv at {meta_csv}")
    else:
        with open(meta_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != METADATA_CSV_COLUMNS:
                errors.append(
                    f"metadata.csv columns mismatch: got {reader.fieldnames}"
                )
            csv_rows = list(reader)
            csv_ids = [row["sample_id"] for row in csv_rows]
            if csv_ids != expected_ids:
                errors.append(
                    f"metadata.csv sample_id mismatch (count={len(csv_ids)} vs {len(expected_ids)})"
                )

    # 2. Check provenance_manifest.csv
    prov_csv = data_dir / "provenance_manifest.csv"
    if not prov_csv.exists():
        errors.append(f"Missing provenance_manifest.csv at {prov_csv}")
    else:
        with open(prov_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != PROVENANCE_MANIFEST_COLUMNS:
                errors.append(
                    f"provenance_manifest.csv columns mismatch: got {reader.fieldnames}"
                )
            prov_rows = list(reader)
            prov_ids = [row["sample_id"] for row in prov_rows]
            if prov_ids != expected_ids:
                errors.append(
                    f"provenance_manifest.csv sample_id mismatch (count={len(prov_ids)} vs {len(expected_ids)})"
                )

    # 3. Check label_studio_tasks.json
    ls_json = data_dir / "label_studio_tasks.json"
    if not ls_json.exists():
        errors.append(f"Missing label_studio_tasks.json at {ls_json}")
    else:
        tasks = json.loads(ls_json.read_text(encoding="utf-8"))
        ls_ids = sorted(t.get("data", {}).get("sample_id", "") for t in tasks)
        if ls_ids != expected_ids:
            errors.append(
                f"label_studio_tasks.json sample_id mismatch (count={len(ls_ids)} vs {len(expected_ids)})"
            )

    return errors


def validate_full_dataset(
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute all validation checks across the dataset directory and return a structured report.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    base_dir = Path(base_dir)

    data_dir = base_dir / "data"
    audio_dir = data_dir / "demo_audio"
    jsonl_path = data_dir / "training_data.jsonl"
    schema_path = base_dir / "configs" / "dataset_schema.json"
    xml_path = base_dir / "configs" / "annotation_config.xml"

    records = load_from_jsonl(jsonl_path)

    schema_errors = validate_records_against_schema(records, schema_path)
    uniqueness_errors = validate_uniqueness_and_duplicates(records)
    audio_errors = validate_audio_files(records, audio_dir)
    semantic_errors = validate_provenance_and_semantics(records)
    split_errors, split_summary = validate_splits_and_leakage(records)
    correspondence_errors = validate_exported_files_correspondence(records, data_dir)

    xml_errors: List[str] = []
    try:
        validate_annotation_config_xml(xml_path, schema_path)
    except Exception as exc:
        xml_errors.append(str(exc))

    all_errors = (
        schema_errors
        + uniqueness_errors
        + audio_errors
        + semantic_errors
        + split_errors
        + correspondence_errors
        + xml_errors
    )

    review_counts = Counter(r["human_review_status"] for r in records)
    category_counts = Counter(r["sound_event_category"] for r in records)
    emotion_counts = Counter(r["primary_emotion"] for r in records)

    report = {
        "valid": len(all_errors) == 0,
        "total_records": len(records),
        "human_reviewed_count": sum(
            1
            for r in records
            if r["human_review_status"] in ("human_reviewed_verified", "human_reviewed_corrected")
        ),
        "review_status_breakdown": dict(review_counts),
        "category_distribution": dict(category_counts),
        "primary_emotion_distribution": dict(emotion_counts),
        "split_summary": split_summary,
        "error_count": len(all_errors),
        "errors": all_errors,
    }
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate ZMH Horror Sound Semantics dataset.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to horror-sound-semantics directory",
    )
    args = parser.parse_args()
    report = validate_full_dataset(args.base_dir)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
