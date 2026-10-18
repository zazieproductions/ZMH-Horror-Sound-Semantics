"""
Dataset Export Module for ZMH Horror Sound Semantics (v0.1).

Exports validated dataset records to:
1. `data/training_data.jsonl` (Full nested schema records, 1 JSON object per line)
2. `data/metadata.csv` (Tabular metadata with all required semantic and signal columns)
3. `data/provenance_manifest.csv` (Comprehensive IP, copyright, and provenance manifest)
4. `data/label_studio_tasks.json` (Label Studio import tasks with provisional predictions)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .label_studio_integration import export_label_studio_tasks

logger = logging.getLogger(__name__)

METADATA_CSV_COLUMNS: List[str] = [
    "sample_id",
    "audio_filename",
    "duration",
    "sample_rate",
    "primary_emotion",
    "secondary_emotions",
    "acoustic_texture",
    "sound_event_category",
    "intensity",
    "intensity_score",
    "tension",
    "tension_score",
    "rhythmic_characteristics",
    "tonality",
    "perceived_spatial_properties",
    "cinematic_usage",
    "natural_language_caption",
    "annotation_confidence",
    "annotation_provenance",
    "human_review_status",
    "copyright_and_licensing_status",
    "split",
    "recipe_id",
    "variant_index",
    "random_seed",
    "generation_method",
    "sha256_checksum",
    "rms_energy",
    "spectral_centroid_hz",
    "spectral_flatness",
    "crest_factor_db",
]

PROVENANCE_MANIFEST_COLUMNS: List[str] = [
    "sample_id",
    "audio_filename",
    "sha256_checksum",
    "generation_method",
    "dsp_engine_name",
    "recipe_id",
    "random_seed",
    "uses_pretrained_ai_model",
    "uses_third_party_samples",
    "is_authentic_human_or_field_recording",
    "waveform_ip_category",
    "acoustic_features_ip_category",
    "schema_and_code_ip_category",
    "curation_and_selection_ip_category",
    "upstream_platform_ip_category",
    "annotation_provenance",
    "human_review_status",
    "copyright_and_licensing_status",
    "commercial_licensing_eligibility",
]


def export_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """Write dataset records to JSON Lines (.jsonl) format sorted by sample_id."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: r["sample_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in sorted_records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")

    logger.info("Exported %d records to JSONL: %s", len(sorted_records), output_path)
    return output_path


def load_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read dataset records from a JSONL file."""
    jsonl_path = Path(jsonl_path)
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {jsonl_path}: {exc}") from exc
    return records


def export_to_metadata_csv(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """Export flattened dataset metadata to CSV format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: r["sample_id"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_CSV_COLUMNS)
        writer.writeheader()
        for rec in sorted_records:
            feats = rec.get("acoustic_features", {})
            row = {
                "sample_id": rec["sample_id"],
                "audio_filename": rec["audio_filename"],
                "duration": rec["duration"],
                "sample_rate": rec["sample_rate"],
                "primary_emotion": rec["primary_emotion"],
                "secondary_emotions": ";".join(rec["secondary_emotions"]),
                "acoustic_texture": rec["acoustic_texture"],
                "sound_event_category": rec["sound_event_category"],
                "intensity": rec["intensity"],
                "intensity_score": rec["intensity_score"],
                "tension": rec["tension"],
                "tension_score": rec["tension_score"],
                "rhythmic_characteristics": rec["rhythmic_characteristics"],
                "tonality": rec["tonality"],
                "perceived_spatial_properties": rec["perceived_spatial_properties"],
                "cinematic_usage": ";".join(rec["cinematic_usage"]),
                "natural_language_caption": rec["natural_language_caption"],
                "annotation_confidence": rec["annotation_confidence"],
                "annotation_provenance": rec["annotation_provenance"],
                "human_review_status": rec["human_review_status"],
                "copyright_and_licensing_status": rec["copyright_and_licensing_status"],
                "split": rec["split"],
                "recipe_id": rec["recipe_id"],
                "variant_index": rec["variant_index"],
                "random_seed": rec["random_seed"],
                "generation_method": rec["generation_method"],
                "sha256_checksum": rec["sha256_checksum"],
                "rms_energy": feats.get("rms_energy", 0.0),
                "spectral_centroid_hz": feats.get("spectral_centroid_hz", 0.0),
                "spectral_flatness": feats.get("spectral_flatness", 0.0),
                "crest_factor_db": feats.get("crest_factor_db", 0.0),
            }
            writer.writerow(row)

    logger.info("Exported %d records to CSV: %s", len(sorted_records), output_path)
    return output_path


def export_to_provenance_manifest_csv(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Export the IP and rights provenance manifest CSV separating:
    - Category A: Third-party software (Label Studio under Apache 2.0)
    - Category B: Zazie-created software & schema
    - Category C: Zazie-owned/controlled compilation, curation & verified human descriptions
    - Category D: Noncopyrightable factual measurements & procedural synthetic audio outputs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: r["sample_id"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_MANIFEST_COLUMNS)
        writer.writeheader()
        for rec in sorted_records:
            synth = rec.get("synthesis_parameters", {})
            row = {
                "sample_id": rec["sample_id"],
                "audio_filename": rec["audio_filename"],
                "sha256_checksum": rec["sha256_checksum"],
                "generation_method": rec["generation_method"],
                "dsp_engine_name": synth.get("engine_name", "unknown"),
                "recipe_id": rec["recipe_id"],
                "random_seed": rec["random_seed"],
                "uses_pretrained_ai_model": rec["uses_pretrained_ai_model"],
                "uses_third_party_samples": rec["uses_third_party_samples"],
                "is_authentic_human_or_field_recording": rec["is_authentic_human_or_field_recording"],
                "waveform_ip_category": "Category_D_Procedural_Synthetic_Waveform_Uncertain_Standalone_Copyright",
                "acoustic_features_ip_category": "Category_D_Factual_Signal_Measurements_Noncopyrightable",
                "schema_and_code_ip_category": "Category_B_Zazie_Original_Software_And_Taxonomy",
                "curation_and_selection_ip_category": "Category_C_Zazie_Controlled_Dataset_Compilation_And_Parameter_Curation",
                "upstream_platform_ip_category": "Category_A_Label_Studio_Apache_2_0_Heartex_HumanSignal",
                "annotation_provenance": rec["annotation_provenance"],
                "human_review_status": rec["human_review_status"],
                "copyright_and_licensing_status": rec["copyright_and_licensing_status"],
                "commercial_licensing_eligibility": (
                    "Licensable_Compilation_And_Data_Delivery_With_Procedural_Audio_Disclosure"
                ),
            }
            writer.writerow(row)

    logger.info("Exported %d records to provenance manifest: %s", len(sorted_records), output_path)
    return output_path


def export_all_dataset_artifacts(
    records: List[Dict[str, Any]],
    data_dir: Path,
) -> Dict[str, str]:
    """Export JSONL, metadata CSV, provenance manifest CSV, and Label Studio tasks JSON."""
    data_dir = Path(data_dir)
    jsonl_path = export_to_jsonl(records, data_dir / "training_data.jsonl")
    csv_path = export_to_metadata_csv(records, data_dir / "metadata.csv")
    prov_path = export_to_provenance_manifest_csv(records, data_dir / "provenance_manifest.csv")
    ls_path = data_dir / "label_studio_tasks.json"
    export_label_studio_tasks(records, ls_path)
    return {
        "jsonl_path": str(jsonl_path),
        "metadata_csv_path": str(csv_path),
        "provenance_csv_path": str(prov_path),
        "label_studio_tasks_path": str(ls_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export dataset artifacts to CSV, JSONL, and Label Studio JSON.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=base_dir / "data" / "training_data.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base_dir / "data",
        help="Output data directory",
    )
    args = parser.parse_args()
    records = load_from_jsonl(args.input_jsonl)
    paths = export_all_dataset_artifacts(records, args.data_dir)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
