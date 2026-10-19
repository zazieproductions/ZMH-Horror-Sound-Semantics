"""
Hugging Face Hub export for ZMH Horror Sound Semantics (v0.1).

Converts the committed dataset (``data/training_data.jsonl`` + ``data/demo_audio/*.wav`` +
``data/provenance_manifest.csv``) into a Hub-compatible **AudioFolder** layout::

    huggingface/
    ├── README.md                     # hand-maintained dataset card (YAML front matter + card body)
    ├── LICENSE.md                    # hand-maintained licensing status notice
    ├── .gitattributes                # generated: Git LFS rules for the Hub
    ├── MANIFEST.sha256               # generated: SHA-256 of every exported file
    ├── data/
    │   ├── train/metadata.jsonl      # generated: one row per clip, ``file_name`` -> WAV in same dir
    │   ├── train/*.wav               # copied from data/demo_audio, SHA-256 verified before and after
    │   ├── validation/...
    │   └── test/...
    └── supplementary/                # generated copies: provenance manifest, schema, IP framework, LS config

Hard rules enforced here (they mirror the dataset's governance requirements):

* Only records with ``generation_method == "procedural_dsp_synthesis"`` and all three provenance
  flags (``uses_pretrained_ai_model``, ``uses_third_party_samples``,
  ``is_authentic_human_or_field_recording``) set to ``False`` are exported. Anything else — in
  particular third-party imports under ``data/third_party/`` — is never copied into the export.
* Every field that distinguishes original ZMH material, algorithmically generated metadata, and
  third-party licensed content is preserved verbatim (``annotation_provenance``,
  ``human_review_status``, ``copyright_and_licensing_status``, ``generation_method``, the three
  provenance booleans, and the per-sample IP categories joined from ``provenance_manifest.csv``).
* ``README.md`` is never generated or overwritten; it is validated instead (split counts, feature
  list, licence placeholder, and a guard against describing unreviewed labels as expert-validated).

Usage::

    python3 -m src.export_huggingface build            # build + validate the export directory
    python3 -m src.export_huggingface verify           # validate an existing export only
    python3 -m src.export_huggingface dataset-info     # print the YAML ``dataset_info`` block
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .export_dataset import load_from_jsonl

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "huggingface"

SPLITS: Tuple[str, ...] = ("train", "validation", "test")
EXPECTED_TOTAL = 200
EXPECTED_SPLIT_COUNTS = {"train": 120, "validation": 40, "test": 40}

# Only this content class is approved for export. Third-party imports are never included.
CONTENT_ORIGIN_ZMH_PROCEDURAL = "zmh_original_procedural_dsp_audio"

# Columns of provenance_manifest.csv that are joined into every Hub row (per-sample IP classes).
PROVENANCE_JOIN_COLUMNS: Tuple[str, ...] = (
    "dsp_engine_name",
    "waveform_ip_category",
    "acoustic_features_ip_category",
    "schema_and_code_ip_category",
    "curation_and_selection_ip_category",
    "upstream_platform_ip_category",
    "commercial_licensing_eligibility",
)

# Top-level training_data.jsonl fields exported verbatim, in this order (``file_name`` is prepended).
TOP_LEVEL_FIELDS: Tuple[str, ...] = (
    "sample_id",
    "audio_filename",
    "dataset_name",
    "dataset_version",
    "publisher",
    "split",
    "sound_event_category",
    "natural_language_caption",
    "primary_emotion",
    "secondary_emotions",
    "acoustic_texture",
    "intensity",
    "intensity_score",
    "tension",
    "tension_score",
    "rhythmic_characteristics",
    "tonality",
    "perceived_spatial_properties",
    "cinematic_usage",
    "duration",
    "sample_rate",
    "sha256_checksum",
    "recipe_id",
    "variant_index",
    "random_seed",
    "generation_method",
    "uses_pretrained_ai_model",
    "uses_third_party_samples",
    "is_authentic_human_or_field_recording",
    "annotation_confidence",
    "annotation_provenance",
    "human_review_status",
    "copyright_and_licensing_status",
    "psychological_label_disclaimer",
)

ACOUSTIC_FEATURE_KEYS: Tuple[str, ...] = (
    "rms_energy",
    "peak_amplitude",
    "crest_factor_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_85_hz",
    "spectral_flatness",
    "low_frequency_energy_ratio",
    "high_frequency_energy_ratio",
    "onset_strength_std",
    "temporal_centroid_norm",
)

SYNTHESIS_SCALAR_KEYS: Tuple[str, ...] = (
    "engine_name",
    "recipe_id",
    "random_seed",
    "base_frequency_hz",
    "modulation_rate_hz",
    "modulation_depth",
    "filter_type",
    "filter_cutoff_hz",
    "noise_mix_ratio",
    "nonlinear_drive",
    "reverb_decay_sec",
    "envelope_attack_sec",
    "envelope_release_sec",
)

SUPPLEMENTARY_FILES: Tuple[Tuple[str, str], ...] = (
    ("data/provenance_manifest.csv", "supplementary/provenance_manifest.csv"),
    ("configs/dataset_schema.json", "supplementary/dataset_schema.json"),
    ("configs/annotation_config.xml", "supplementary/label_studio_annotation_config.xml"),
    ("docs/IP_OWNERSHIP.md", "supplementary/IP_OWNERSHIP.md"),
    ("docs/ANNOTATION_GUIDELINES.md", "supplementary/ANNOTATION_GUIDELINES.md"),
)

GITATTRIBUTES = """# Git LFS rules for the Hugging Face Hub dataset repository (generated by src/export_huggingface.py)
*.wav filter=lfs diff=lfs merge=lfs -text
*.flac filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.ogg filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
"""

# Public redistribution licences that must NOT be set in the card without explicit owner approval.
PUBLIC_REDISTRIBUTION_LICENSES = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-nd-4.0",
    "cc-by-3.0",
    "cc-by-sa-3.0",
    "cc",
    "mit",
    "apache-2.0",
    "odc-by",
    "odbl",
    "pddl",
    "gpl-3.0",
    "lgpl-3.0",
    "bsd-3-clause",
    "unlicense",
    "openrail",
    "cdla-sharing-1.0",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
    "etalab-2.0",
    "afl-3.0",
    "artistic-2.0",
}

# Affirmative wording that would misrepresent unreviewed algorithmic labels. The card may only use
# these phrases in a negated form ("not expert-validated", "no expert-validated ...").
FORBIDDEN_AFFIRMATIVE_PHRASES = (
    r"expert[- ]validated",
    r"expert[- ]verified",
    r"expert[- ]reviewed",
    r"human[- ]validated",
    r"human[- ]verified",
    r"professionally (?:validated|verified|annotated)",
    r"validated by (?:audio )?experts",
)
NEGATION_PREFIXES = (
    "not ",
    "no ",
    "non-",
    "never ",
    "zero ",
    "none ",
    "are not ",
    "is not ",
    "be ",
    "been ",
    "without ",
    "rather than ",
    "instead of ",
    "as ",
    "not yet ",
    "0 ",
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def is_exportable_zmh_original(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, reason). Only original ZMH procedural audio with clean provenance flags passes."""
    if rec.get("generation_method") != "procedural_dsp_synthesis":
        return False, f"generation_method={rec.get('generation_method')!r} is not procedural_dsp_synthesis"
    for flag in ("uses_pretrained_ai_model", "uses_third_party_samples", "is_authentic_human_or_field_recording"):
        if rec.get(flag) is not False:
            return False, f"{flag}={rec.get(flag)!r} (must be False for export)"
    if "third_party" in str(rec.get("audio_filename", "")).replace("\\", "/"):
        return False, "audio path points at a third-party import"
    return True, ""


def build_hf_row(rec: Dict[str, Any], provenance: Dict[str, str]) -> Dict[str, Any]:
    """Project one training_data.jsonl record onto a flat, Arrow-friendly Hub row."""
    row: Dict[str, Any] = {"file_name": rec["audio_filename"]}
    for key in TOP_LEVEL_FIELDS:
        row[key] = rec[key]

    # Explicit content-origin marker. Third-party licensed content is never merged into this config;
    # if it is ever approved it ships as a separate config with its own attribution columns.
    row["content_origin"] = CONTENT_ORIGIN_ZMH_PROCEDURAL

    # Per-sample IP classification joined from provenance_manifest.csv (Category A/B/C/D framework).
    for col in PROVENANCE_JOIN_COLUMNS:
        row[col] = provenance[col]

    synth = rec["synthesis_parameters"]
    row["synthesis_parameters"] = {k: synth[k] for k in SYNTHESIS_SCALAR_KEYS}
    # ``additional_params`` keys differ per DSP engine; serialise to JSON so the Arrow schema is stable.
    row["synthesis_parameters"]["additional_params_json"] = json.dumps(
        synth.get("additional_params", {}), sort_keys=True, ensure_ascii=False
    )

    feats = rec["acoustic_features"]
    row["acoustic_features"] = {k: float(feats[k]) for k in ACOUSTIC_FEATURE_KEYS}

    row["temporal_events"] = [
        {"start_sec": float(ev["start_sec"]), "end_sec": float(ev["end_sec"]), "event_label": str(ev["event_label"])}
        for ev in rec["temporal_events"]
    ]
    return row


def _read_wav_header(path: Path) -> Tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def build_huggingface_export(
    base_dir: Path = BASE_DIR,
    output_dir: Optional[Path] = None,
    *,
    clean: bool = True,
) -> Dict[str, Any]:
    """Assemble the Hub-ready folder and return a build report."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir / "huggingface"
    audio_dir = base_dir / "data" / "demo_audio"
    records = load_from_jsonl(base_dir / "data" / "training_data.jsonl")
    provenance = load_provenance_manifest(base_dir / "data" / "provenance_manifest.csv")

    # 1. Governance gate: refuse anything that is not original ZMH procedural audio.
    rejected: List[str] = []
    for rec in records:
        ok, reason = is_exportable_zmh_original(rec)
        if not ok:
            rejected.append(f"{rec.get('sample_id')}: {reason}")
        if rec["sample_id"] not in provenance:
            rejected.append(f"{rec.get('sample_id')}: missing from provenance_manifest.csv")
    if rejected:
        raise ValueError(
            "Refusing to build Hugging Face export: records that are not original ZMH procedural audio "
            "(or lack provenance rows) were found. Third-party content requires separate approval.\n  "
            + "\n  ".join(rejected)
        )

    # 2. Reset generated parts only. README.md / LICENSE.md are hand-maintained and left untouched.
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for sub in ("data", "supplementary"):
            shutil.rmtree(output_dir / sub, ignore_errors=True)
        for name in (".gitattributes", "MANIFEST.sha256"):
            (output_dir / name).unlink(missing_ok=True)

    # 3. Copy audio + write per-split metadata.jsonl.
    per_split: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
    copied_bytes = 0
    for rec in sorted(records, key=lambda r: r["sample_id"]):
        split = rec["split"]
        if split not in SPLITS:
            raise ValueError(f"{rec['sample_id']}: unknown split {split!r}")
        src = audio_dir / rec["audio_filename"]
        if not src.is_file():
            raise FileNotFoundError(f"{rec['sample_id']}: audio file missing: {src}")
        actual = sha256_of_file(src)
        if actual != rec["sha256_checksum"]:
            raise ValueError(
                f"{rec['sample_id']}: SHA-256 mismatch before copy ({actual} != {rec['sha256_checksum']})"
            )
        split_dir = output_dir / "data" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dst = split_dir / rec["audio_filename"]
        shutil.copyfile(src, dst)
        if sha256_of_file(dst) != rec["sha256_checksum"]:
            raise ValueError(f"{rec['sample_id']}: SHA-256 mismatch after copy")
        copied_bytes += dst.stat().st_size
        per_split[split].append(build_hf_row(rec, provenance[rec["sample_id"]]))

    for split in SPLITS:
        split_dir = output_dir / "data" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for row in per_split[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4. Supplementary provenance / schema material (copied verbatim; third_party/ is never included).
    for rel_src, rel_dst in SUPPLEMENTARY_FILES:
        src = base_dir / rel_src
        dst = output_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    (output_dir / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    # 5. Manifest of everything in the export (README/LICENSE included so the Hub copy can be audited).
    manifest_lines = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        manifest_lines.append(f"{sha256_of_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    report = {
        "output_dir": str(output_dir),
        "records_exported": sum(len(v) for v in per_split.values()),
        "records_rejected": 0,
        "counts_by_split": {s: len(per_split[s]) for s in SPLITS},
        "audio_bytes": copied_bytes,
        "files_in_manifest": len(manifest_lines),
    }
    logger.info("Hugging Face export built: %s", json.dumps(report))
    return report


# --------------------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------------------
def parse_readme_front_matter(readme_path: Path) -> Tuple[Dict[str, Any], str]:
    """Return (yaml_dict, body). Requires PyYAML."""
    import yaml  # local import: only needed for verification

    text = readme_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not match:
        raise ValueError(f"{readme_path}: missing YAML front matter (--- ... ---)")
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def _check_card_wording(body: str) -> List[str]:
    """Flag affirmative 'expert-validated'-style claims that are not negated."""
    errors: List[str] = []
    lowered = body.lower()
    for pattern in FORBIDDEN_AFFIRMATIVE_PHRASES:
        for m in re.finditer(pattern, lowered):
            prefix = lowered[max(0, m.start() - 24) : m.start()]
            if not any(prefix.rstrip().endswith(neg.strip()) or neg in prefix for neg in NEGATION_PREFIXES):
                errors.append(
                    f"README body contains an affirmative claim '{m.group(0)}' near: "
                    f"...{lowered[max(0, m.start() - 40) : m.end() + 40]!r}"
                )
    return errors


def normalize_features_yaml(features: Any) -> Any:
    """Normalise the legacy ``sequence:`` YAML dialect to the canonical ``list:`` dialect for comparison."""
    if isinstance(features, list):
        return [normalize_features_yaml(f) for f in features]
    if isinstance(features, dict):
        out: Dict[str, Any] = {}
        for key, value in features.items():
            out["list" if key == "sequence" else key] = normalize_features_yaml(value)
        return out
    return features


def verify_readme(
    output_dir: Path, counts_by_split: Dict[str, int], features_yaml: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    errors: List[str] = []
    readme = output_dir / "README.md"
    if not readme.is_file():
        return [f"missing {readme}"]
    meta, body = parse_readme_front_matter(readme)

    # Licence: must remain a non-public placeholder until the owner approves one.
    license_value = str(meta.get("license", "")).lower()
    if not license_value:
        errors.append("README front matter has no 'license' field")
    elif license_value in PUBLIC_REDISTRIBUTION_LICENSES:
        errors.append(
            f"README selects public redistribution licence '{license_value}' — requires explicit owner approval"
        )
    elif license_value == "other" and not meta.get("license_name"):
        errors.append("license: other requires license_name")

    # configs → data_files must reference the split folders that actually exist.
    configs = meta.get("configs") or []
    if not configs:
        errors.append("README front matter has no 'configs' block")
    else:
        declared = {df["split"]: df["path"] for df in configs[0].get("data_files", [])}
        for split in SPLITS:
            if split not in declared:
                errors.append(f"configs.data_files lacks split '{split}'")
            elif not declared[split].startswith(f"data/{split}/"):
                errors.append(
                    f"configs.data_files for '{split}' points at {declared[split]!r}, expected data/{split}/..."
                )

    # dataset_info.splits must match the real counts.
    info = meta.get("dataset_info") or {}
    for split_info in info.get("splits", []):
        name = split_info.get("name")
        if name in counts_by_split and split_info.get("num_examples") != counts_by_split[name]:
            errors.append(
                f"dataset_info.splits[{name}].num_examples={split_info.get('num_examples')} != {counts_by_split[name]}"
            )
    if features_yaml is not None and info.get("features") is not None:
        if normalize_features_yaml(info["features"]) != normalize_features_yaml(features_yaml):
            errors.append(
                "dataset_info.features in README differ from the features inferred by `datasets` (run `dataset-info`)"
            )

    # Wording guard + mandatory provenance disclosures.
    errors.extend(_check_card_wording(body))
    for needle in (
        "unreviewed_provisional",
        "algorithmic_dsp_and_acoustic_heuristic_v0.1",
        "procedural_dsp_synthesis",
    ):
        if needle not in body:
            errors.append(f"README body must mention '{needle}'")
    if "third" not in body.lower() or "party" not in body.lower():
        errors.append("README body must state the third-party content position")
    return errors


def verify_huggingface_export(
    base_dir: Path = BASE_DIR,
    output_dir: Optional[Path] = None,
    *,
    load_with_datasets: bool = True,
    decode_audio: bool = True,
) -> Dict[str, Any]:
    """Cross-check the export folder against the source dataset (and optionally load it with `datasets`)."""
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir / "huggingface"
    errors: List[str] = []

    source = {r["sample_id"]: r for r in load_from_jsonl(base_dir / "data" / "training_data.jsonl")}

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    wav_count = 0
    seen_ids: Counter = Counter()
    for split in SPLITS:
        split_dir = output_dir / "data" / split
        meta_path = split_dir / "metadata.jsonl"
        if not meta_path.is_file():
            errors.append(f"missing {meta_path}")
            continue
        rows = load_from_jsonl(meta_path)
        rows_by_split[split] = rows
        wavs = {p.name for p in split_dir.glob("*.wav")}
        wav_count += len(wavs)
        names_in_meta = {r["file_name"] for r in rows}
        if wavs != names_in_meta:
            errors.append(
                f"{split}: WAV files and metadata rows differ: only_on_disk={sorted(wavs - names_in_meta)[:5]} "
                f"only_in_meta={sorted(names_in_meta - wavs)[:5]}"
            )
        for row in rows:
            sid = row["sample_id"]
            seen_ids[sid] += 1
            src = source.get(sid)
            if src is None:
                errors.append(f"{split}/{sid}: not present in training_data.jsonl")
                continue
            if row["split"] != split:
                errors.append(f"{sid}: row.split={row['split']} but stored under data/{split}/")
            if row["file_name"] != src["audio_filename"]:
                errors.append(f"{sid}: file_name {row['file_name']} != {src['audio_filename']}")
            wav_path = split_dir / row["file_name"]
            if wav_path.is_file():
                if sha256_of_file(wav_path) != src["sha256_checksum"]:
                    errors.append(f"{sid}: exported WAV SHA-256 mismatch")
                try:
                    sr, ch, width, frames = _read_wav_header(wav_path)
                    if (sr, ch, width) != (src["sample_rate"], 1, 2):
                        errors.append(f"{sid}: WAV header {sr}Hz/{ch}ch/{width * 8}bit unexpected")
                    if abs(frames / sr - src["duration"]) > 1e-3:
                        errors.append(f"{sid}: WAV duration {frames / sr:.3f}s != {src['duration']}")
                except wave.Error as exc:
                    errors.append(f"{sid}: unreadable WAV: {exc}")
            # Provenance / review-status fields must be carried over verbatim.
            for key in (
                "annotation_provenance",
                "human_review_status",
                "copyright_and_licensing_status",
                "generation_method",
                "uses_pretrained_ai_model",
                "uses_third_party_samples",
                "is_authentic_human_or_field_recording",
                "sha256_checksum",
                "natural_language_caption",
                "primary_emotion",
                "sound_event_category",
                "recipe_id",
                "variant_index",
                "random_seed",
            ):
                if row.get(key) != src[key]:
                    errors.append(f"{sid}: field {key} altered in export ({row.get(key)!r} != {src[key]!r})")
            if row.get("content_origin") != CONTENT_ORIGIN_ZMH_PROCEDURAL:
                errors.append(f"{sid}: content_origin={row.get('content_origin')!r}")
            if row.get("uses_third_party_samples") is not False or row.get("third_party_source_dataset"):
                errors.append(f"{sid}: third-party content marker present — third-party content is not approved")
            if len(row["temporal_events"]) != len(src["temporal_events"]):
                errors.append(f"{sid}: temporal_events count differs")
            if json.loads(row["synthesis_parameters"]["additional_params_json"]) != src["synthesis_parameters"].get(
                "additional_params", {}
            ):
                errors.append(f"{sid}: additional_params altered")

    total_rows = sum(len(v) for v in rows_by_split.values())
    duplicates = [sid for sid, n in seen_ids.items() if n > 1]
    if duplicates:
        errors.append(f"duplicate sample_ids across splits: {duplicates[:5]}")
    missing = sorted(set(source) - set(seen_ids))
    if missing:
        errors.append(f"{len(missing)} source records missing from export, e.g. {missing[:5]}")
    counts_by_split = {s: len(rows_by_split.get(s, [])) for s in SPLITS}
    if counts_by_split != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts {counts_by_split} != expected {EXPECTED_SPLIT_COUNTS}")
    if wav_count != EXPECTED_TOTAL:
        errors.append(f"{wav_count} WAV files exported, expected {EXPECTED_TOTAL}")
    if (output_dir / "data" / "third_party").exists() or any(output_dir.rglob("third_party")):
        errors.append("export contains a third_party directory — not approved")

    # MANIFEST.sha256 must describe the on-disk files exactly (README/LICENSE included).
    manifest_path = output_dir / "MANIFEST.sha256"
    if manifest_path.is_file():
        listed: Dict[str, str] = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, rel = line.split("  ", 1)
                listed[rel] = digest
        on_disk = {
            p.relative_to(output_dir).as_posix(): p
            for p in output_dir.rglob("*")
            if p.is_file() and p.name != "MANIFEST.sha256"
        }
        if set(listed) != set(on_disk):
            errors.append(
                f"MANIFEST.sha256 file set differs from disk: extra={sorted(set(listed) - set(on_disk))[:5]} "
                f"missing={sorted(set(on_disk) - set(listed))[:5]}"
            )
        for rel, digest in listed.items():
            if rel in on_disk and rel not in ("README.md", "LICENSE.md") and sha256_of_file(on_disk[rel]) != digest:
                errors.append(f"MANIFEST.sha256 digest mismatch for {rel}")
    else:
        errors.append("missing MANIFEST.sha256")

    # Optional: load through the `datasets` AudioFolder builder exactly as the Hub would.
    datasets_report: Dict[str, Any] = {"attempted": False}
    features_yaml: Optional[List[Dict[str, Any]]] = None
    if load_with_datasets:
        datasets_report, features_yaml, ds_errors = _load_with_datasets(output_dir, decode_audio=decode_audio)
        errors.extend(ds_errors)

    errors.extend(verify_readme(output_dir, counts_by_split, features_yaml))

    report = {
        "valid": not errors,
        "output_dir": str(output_dir),
        "total_rows": total_rows,
        "wav_files": wav_count,
        "counts_by_split": counts_by_split,
        "human_review_status_breakdown": dict(
            Counter(r["human_review_status"] for rows in rows_by_split.values() for r in rows)
        ),
        "annotation_provenance_breakdown": dict(
            Counter(r["annotation_provenance"] for rows in rows_by_split.values() for r in rows)
        ),
        "content_origin_breakdown": dict(
            Counter(r.get("content_origin") for rows in rows_by_split.values() for r in rows)
        ),
        "third_party_records": sum(
            1
            for rows in rows_by_split.values()
            for r in rows
            if r.get("uses_third_party_samples") or r.get("third_party_source_dataset")
        ),
        "datasets_library": datasets_report,
        "error_count": len(errors),
        "errors": errors,
    }
    return report


def _load_with_datasets(
    output_dir: Path, *, decode_audio: bool
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]], List[str]]:
    """Load the export with the `datasets` AudioFolder builder and decode a sample from every split."""
    errors: List[str] = []
    try:
        import datasets  # noqa: WPS433
    except ImportError:
        return {"attempted": False, "reason": "datasets not installed"}, None, errors

    try:
        ds = datasets.load_dataset("audiofolder", data_dir=str(output_dir / "data"))
    except Exception as exc:  # pragma: no cover - surfaced as a verification error
        return (
            {"attempted": True, "loaded": False, "error": f"{type(exc).__name__}: {exc}"},
            None,
            [f"datasets.load_dataset failed: {exc}"],
        )

    report: Dict[str, Any] = {"attempted": True, "loaded": True, "version": datasets.__version__, "splits": {}}
    features_yaml = None
    for split in SPLITS:
        if split not in ds:
            errors.append(f"datasets: split '{split}' missing")
            continue
        d = ds[split]
        if features_yaml is None:
            features_yaml = d.features._to_yaml_list()
        if "audio" not in d.features or not isinstance(d.features["audio"], datasets.Audio):
            errors.append(f"datasets: '{split}' has no Audio feature")
        if "file_name" in d.features:
            errors.append(f"datasets: '{split}' still exposes file_name (metadata not associated)")
        split_report: Dict[str, Any] = {"num_rows": d.num_rows, "num_columns": len(d.features)}
        if decode_audio and d.num_rows:
            try:
                decoded = _decode_first_audio(d)
                split_report.update(decoded)
                if decoded["sample_rate"] != d[0]["sample_rate"] or decoded["num_samples"] <= 0:
                    errors.append(f"datasets: '{split}' decoded audio inconsistent: {decoded}")
            except Exception as exc:  # pragma: no cover
                errors.append(f"datasets: audio decode failed for '{split}': {type(exc).__name__}: {exc}")
        report["splits"][split] = split_report
    if features_yaml is not None:
        report["num_features"] = len(features_yaml)
    return report, features_yaml, errors


def _decode_first_audio(d: Any) -> Dict[str, Any]:
    """Decode row 0's audio using whichever backend this `datasets` version provides."""
    audio = d[0]["audio"]
    if isinstance(audio, dict) and "array" in audio:  # datasets < 4: soundfile backend
        return {
            "decoder": "datasets/soundfile",
            "sample_rate": int(audio["sampling_rate"]),
            "num_samples": int(len(audio["array"])),
        }
    try:  # datasets >= 4: torchcodec AudioDecoder
        samples = audio.get_all_samples()
        return {
            "decoder": "datasets/torchcodec",
            "sample_rate": int(samples.sample_rate),
            "num_samples": int(samples.data.shape[-1]),
        }
    except Exception:
        # Fall back to decoding the raw file ourselves (e.g. FFmpeg missing locally).
        import soundfile as sf

        raw = d.cast_column("audio", __import__("datasets").Audio(decode=False))[0]["audio"]
        data, sr = sf.read(raw["path"]) if raw.get("path") else sf.read(__import__("io").BytesIO(raw["bytes"]))
        return {"decoder": "soundfile-fallback", "sample_rate": int(sr), "num_samples": int(len(data))}


def print_dataset_info_yaml(output_dir: Path) -> str:
    """Emit the ``dataset_info`` block (features + split sizes) for pasting into README.md."""
    import datasets
    import yaml

    ds = datasets.load_dataset("audiofolder", data_dir=str(output_dir / "data"))
    first = ds[SPLITS[0]]
    splits = []
    download_size = 0
    for split in SPLITS:
        num_bytes = sum(p.stat().st_size for p in (output_dir / "data" / split).glob("*") if p.is_file())
        download_size += num_bytes
        splits.append({"name": split, "num_bytes": num_bytes, "num_examples": ds[split].num_rows})
    info = {
        "dataset_info": {
            "features": normalize_features_yaml(first.features._to_yaml_list()),
            "splits": splits,
            "download_size": download_size,
            "dataset_size": download_size,
        }
    }
    return yaml.safe_dump(info, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build / verify the Hugging Face AudioFolder export.")
    parser.add_argument("command", nargs="?", default="build", choices=("build", "verify", "dataset-info"))
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to <base-dir>/huggingface")
    parser.add_argument("--skip-datasets-load", action="store_true", help="do not load the export with `datasets`")
    parser.add_argument("--no-decode", action="store_true", help="skip audio decoding during `datasets` check")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.output_dir or (args.base_dir / "huggingface")
    if args.command == "dataset-info":
        print(print_dataset_info_yaml(output_dir))
        return 0
    if args.command == "build":
        build_report = build_huggingface_export(args.base_dir, output_dir)
        print(json.dumps({"build": build_report}, indent=2))
    report = verify_huggingface_export(
        args.base_dir, output_dir, load_with_datasets=not args.skip_datasets_load, decode_audio=not args.no_decode
    )
    print(json.dumps({"verify": report}, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
