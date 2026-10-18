"""
Label Studio Integration Module for ZMH Horror Sound Semantics (v0.1).

Provides:
1. Validation of `configs/annotation_config.xml` against Label Studio's tag and schema rules.
2. Conversion of dataset records into Label Studio task import JSON (`data` + `predictions`)
   for automatic provisional pre-labeling.
3. Human review and correction ingestion workflow (`ingest_human_annotations_from_label_studio`)
   that merges human-submitted `annotations` from Label Studio exports back into the dataset
   while updating provenance (`human_expert_reviewed_v0.1` / `human_expert_corrected_v0.1`)
   and review status (`human_reviewed_verified` / `human_reviewed_corrected`).
4. Optional REST API client (`LabelStudioClient`) for live Label Studio server synchronization
   with idempotent task deduplication.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

SINGLE_CHOICE_FIELDS: List[str] = [
    "sound_event_category",
    "acoustic_texture",
    "primary_emotion",
    "intensity",
    "tension",
    "rhythmic_characteristics",
    "tonality",
    "perceived_spatial_properties",
    "annotation_provenance",
    "human_review_status",
    "copyright_and_licensing_status",
]

MULTI_CHOICE_FIELDS: List[str] = [
    "secondary_emotions",
    "cinematic_usage",
]


def validate_annotation_config_xml(
    xml_path: Path,
    dataset_schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate the Label Studio XML configuration file for structural integrity,
    unique control/object names, valid `toName` references, and alignment with
    `configs/dataset_schema.json` controlled vocabularies.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation config XML not found: {xml_path}")

    xml_text = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    if root.tag != "View":
        raise ValueError(f"Root tag of Label Studio config must be <View>, got <{root.tag}>")

    named_elements: Dict[str, ET.Element] = {}
    to_name_refs: List[Tuple[str, str]] = []
    choices_by_control: Dict[str, List[str]] = {}
    labels_by_control: Dict[str, List[str]] = {}

    for elem in root.iter():
        name = elem.attrib.get("name")
        if name:
            if name in named_elements:
                raise ValueError(f"Duplicate `name` attribute in annotation_config.xml: '{name}'")
            named_elements[name] = elem

        to_name = elem.attrib.get("toName")
        if to_name and name:
            to_name_refs.append((name, to_name))

        if elem.tag == "Choices" and name:
            choices_by_control[name] = [
                c.attrib["value"] for c in elem.findall("Choice") if "value" in c.attrib
            ]
        elif elem.tag == "Labels" and name:
            labels_by_control[name] = [
                l.attrib["value"] for l in elem.findall("Label") if "value" in l.attrib
            ]

    if "audio" not in named_elements or named_elements["audio"].tag != "Audio":
        raise ValueError("annotation_config.xml must define `<Audio name=\"audio\" .../>`")

    for ctrl_name, target_name in to_name_refs:
        for t_part in target_name.split(","):
            if t_part not in named_elements:
                raise ValueError(
                    f"Control '{ctrl_name}' references non-existent toName='{t_part}'"
                )

    required_controls = set(SINGLE_CHOICE_FIELDS + MULTI_CHOICE_FIELDS + [
        "temporal_events",
        "natural_language_caption",
        "annotation_confidence",
    ])
    missing_controls = required_controls - set(named_elements.keys())
    if missing_controls:
        raise ValueError(
            f"annotation_config.xml is missing required controls: {sorted(missing_controls)}"
        )

    # Cross-check against dataset_schema.json enums if provided
    if dataset_schema_path and Path(dataset_schema_path).exists():
        schema = json.loads(Path(dataset_schema_path).read_text(encoding="utf-8"))
        defs = schema.get("$defs", {})
        enum_mapping = {
            "sound_event_category": "sound_event_category_enum",
            "acoustic_texture": "acoustic_texture_enum",
            "primary_emotion": "emotion_enum",
            "secondary_emotions": "emotion_enum",
            "intensity": "intensity_enum",
            "tension": "tension_enum",
            "rhythmic_characteristics": "rhythmic_characteristics_enum",
            "tonality": "tonality_enum",
            "perceived_spatial_properties": "perceived_spatial_properties_enum",
            "cinematic_usage": "cinematic_usage_enum",
            "annotation_provenance": "annotation_provenance_enum",
            "human_review_status": "human_review_status_enum",
            "copyright_and_licensing_status": "copyright_and_licensing_status_enum",
        }
        for control_name, def_key in enum_mapping.items():
            expected_set = set(defs[def_key]["enum"])
            actual_set = set(choices_by_control.get(control_name, []))
            if expected_set != actual_set:
                raise ValueError(
                    f"Vocabulary mismatch for '{control_name}' between XML and JSON schema: "
                    f"expected={sorted(expected_set)}, actual={sorted(actual_set)}"
                )

        if set(labels_by_control.get("temporal_events", [])) != set(
            defs["sound_event_category_enum"]["enum"]
        ):
            raise ValueError("temporal_events <Labels> do not match sound_event_category_enum")

    return {
        "valid": True,
        "named_controls_count": len(named_elements),
        "choice_controls": list(choices_by_control.keys()),
        "label_controls": list(labels_by_control.keys()),
    }


def record_to_label_studio_task(
    record: Dict[str, Any],
    audio_url_prefix: str = "/data/local-files/?d=demo_audio/",
) -> Dict[str, Any]:
    """
    Convert a dataset record into a Label Studio Task object containing `data` and
    pre-populated `predictions` (provisional algorithmic labels).
    """
    sid = record["sample_id"]
    audio_url = f"{audio_url_prefix.rstrip('/')}/{record['audio_filename']}"

    results: List[Dict[str, Any]] = []

    # 1. Temporal sound-event segments on the waveform
    for idx, ev in enumerate(record.get("temporal_events", [])):
        results.append(
            {
                "id": f"{sid}_ev_{idx:02d}",
                "from_name": "temporal_events",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": float(ev["start_sec"]),
                    "end": float(ev["end_sec"]),
                    "labels": [ev["event_label"]],
                },
            }
        )

    # 2. Single-choice semantic & provenance controls
    for field in SINGLE_CHOICE_FIELDS:
        val = record.get(field)
        if val is not None:
            results.append(
                {
                    "id": f"{sid}_{field}",
                    "from_name": field,
                    "to_name": "audio",
                    "type": "choices",
                    "value": {"choices": [str(val)]},
                }
            )

    # 3. Multi-choice semantic controls
    for field in MULTI_CHOICE_FIELDS:
        vals = record.get(field, [])
        if vals:
            results.append(
                {
                    "id": f"{sid}_{field}",
                    "from_name": field,
                    "to_name": "audio",
                    "type": "choices",
                    "value": {"choices": [str(v) for v in vals]},
                }
            )

    # 4. Natural-language caption (TextArea)
    results.append(
        {
            "id": f"{sid}_caption",
            "from_name": "natural_language_caption",
            "to_name": "audio",
            "type": "textarea",
            "value": {"text": [record["natural_language_caption"]]},
        }
    )

    # 5. Annotation confidence (Number)
    results.append(
        {
            "id": f"{sid}_confidence",
            "from_name": "annotation_confidence",
            "to_name": "audio",
            "type": "number",
            "value": {"number": float(record["annotation_confidence"])},
        }
    )

    task: Dict[str, Any] = {
        "data": {
            "audio_url": audio_url,
            "sample_id": record["sample_id"],
            "audio_filename": record["audio_filename"],
            "duration": record["duration"],
            "sample_rate": record["sample_rate"],
            "split": record["split"],
            "recipe_id": record["recipe_id"],
            "random_seed": record["random_seed"],
            "generation_method": record["generation_method"],
            "sha256_checksum": record["sha256_checksum"],
            "psychological_label_disclaimer": record["psychological_label_disclaimer"],
            "synthesis_parameters": record["synthesis_parameters"],
            "acoustic_features": record["acoustic_features"],
        },
        "predictions": [
            {
                "model_version": "zmh-hss-provisional-v0.1",
                "score": float(record["annotation_confidence"]),
                "result": results,
            }
        ],
    }
    return task


def export_label_studio_tasks(
    records: List[Dict[str, Any]],
    output_path: Path,
    audio_url_prefix: str = "/data/local-files/?d=demo_audio/",
) -> List[Dict[str, Any]]:
    """Write Label Studio import tasks JSON file from dataset records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [record_to_label_studio_task(r, audio_url_prefix=audio_url_prefix) for r in records]
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    logger.info("Exported %d Label Studio tasks to %s", len(tasks), output_path)
    return tasks


def ingest_human_annotations_from_label_studio(
    ls_exported_tasks: List[Dict[str, Any]],
    existing_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Human review & correction workflow:
    Reads exported Label Studio tasks containing completed human `annotations` and merges
    verified or corrected human judgments into the dataset records without duplicating items.

    Returns `(updated_records, summary_stats)`.
    """
    records_by_id: Dict[str, Dict[str, Any]] = {
        r["sample_id"]: copy.deepcopy(r) for r in existing_records
    }
    stats = {
        "tasks_processed": 0,
        "human_verified_count": 0,
        "human_corrected_count": 0,
        "rejected_count": 0,
        "unreviewed_count": 0,
    }

    semantic_compare_fields = [
        "sound_event_category",
        "acoustic_texture",
        "primary_emotion",
        "secondary_emotions",
        "intensity",
        "tension",
        "rhythmic_characteristics",
        "tonality",
        "perceived_spatial_properties",
        "cinematic_usage",
        "natural_language_caption",
    ]

    for task in ls_exported_tasks:
        stats["tasks_processed"] += 1
        data = task.get("data", {})
        sid = data.get("sample_id")
        if not sid or sid not in records_by_id:
            continue

        annotations = [
            a for a in task.get("annotations", []) if not a.get("was_cancelled", False)
        ]
        if not annotations:
            continue

        # Take the most recent completed human annotation
        latest_ann = annotations[-1]
        result_items = latest_ann.get("result", [])
        if not result_items:
            continue

        rec = records_by_id[sid]
        original_snapshot = {k: copy.deepcopy(rec.get(k)) for k in semantic_compare_fields}

        extracted_events: List[Dict[str, Any]] = []
        reviewer_notes = ""

        for item in result_items:
            from_name = item.get("from_name")
            rtype = item.get("type")
            val = item.get("value", {})

            if from_name == "temporal_events" and rtype == "labels":
                labels = val.get("labels", [])
                if labels:
                    extracted_events.append(
                        {
                            "start_sec": round(float(val.get("start", 0.0)), 3),
                            "end_sec": round(float(val.get("end", rec["duration"])), 3),
                            "event_label": str(labels[0]),
                        }
                    )
            elif from_name in SINGLE_CHOICE_FIELDS and rtype == "choices":
                choices = val.get("choices", [])
                if choices:
                    rec[from_name] = str(choices[0])
            elif from_name in MULTI_CHOICE_FIELDS and rtype == "choices":
                choices = val.get("choices", [])
                if choices:
                    rec[from_name] = [str(c) for c in choices]
            elif from_name == "natural_language_caption" and rtype == "textarea":
                texts = val.get("text", [])
                if texts and texts[0].strip():
                    rec["natural_language_caption"] = texts[0].strip()
            elif from_name == "reviewer_notes" and rtype == "textarea":
                texts = val.get("text", [])
                if texts:
                    reviewer_notes = texts[0].strip()
            elif from_name == "annotation_confidence" and rtype == "number":
                num = val.get("number")
                if num is not None:
                    rec["annotation_confidence"] = round(float(num), 3)

        if extracted_events:
            rec["temporal_events"] = extracted_events

        # Determine whether the human reviewer modified any semantic field
        was_corrected = any(
            rec.get(k) != original_snapshot.get(k) for k in semantic_compare_fields
        )

        # Ensure secondary_emotions does not contain primary_emotion
        if rec["primary_emotion"] in rec["secondary_emotions"] and len(rec["secondary_emotions"]) > 1:
            rec["secondary_emotions"] = [
                e for e in rec["secondary_emotions"] if e != rec["primary_emotion"]
            ]

        explicit_status = rec.get("human_review_status")
        if explicit_status == "rejected":
            stats["rejected_count"] += 1
        elif was_corrected or explicit_status == "human_reviewed_corrected":
            rec["human_review_status"] = "human_reviewed_corrected"
            rec["annotation_provenance"] = "human_expert_corrected_v0.1"
            stats["human_corrected_count"] += 1
        else:
            rec["human_review_status"] = "human_reviewed_verified"
            rec["annotation_provenance"] = "human_expert_reviewed_v0.1"
            stats["human_verified_count"] += 1

        if rec["human_review_status"] in ("human_reviewed_verified", "human_reviewed_corrected"):
            rec["psychological_label_disclaimer"] = (
                "Affective and semantic labels for this sample have been audited by a human "
                f"reviewer ({rec['human_review_status']}) via Label Studio."
            )

        rec["human_review_audit"] = {
            "reviewed_by": str(latest_ann.get("completed_by", "human_annotator")),
            "reviewed_at": str(
                latest_ann.get("updated_at")
                or latest_ann.get("created_at")
                or "2026-10-18T00:00:00Z"
            ),
            "lead_time_seconds": float(latest_ann.get("lead_time") or 0.0),
            "was_corrected": bool(was_corrected),
            "reviewer_notes": reviewer_notes,
        }

    updated_list = [records_by_id[k] for k in sorted(records_by_id.keys())]
    stats["unreviewed_count"] = sum(
        1 for r in updated_list if r["human_review_status"] == "unreviewed_provisional"
    )
    return updated_list, stats


class LabelStudioClient:
    """
    Lightweight REST API client for configuring and syncing the Horror Sound Semantics
    project with a running Label Studio instance.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
        )

    def check_health(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/api/health", timeout=self.timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def create_or_get_project(
        self,
        title: str,
        label_config_xml: str,
        description: str = "ZMH Horror Sound Semantics v0.1 — Audio & Semantic Annotation Project",
    ) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/api/projects", timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        projects = payload.get("results", payload) if isinstance(payload, dict) else payload
        for proj in projects:
            if proj.get("title") == title:
                return proj

        create_resp = self.session.post(
            f"{self.base_url}/api/projects",
            json={
                "title": title,
                "description": description,
                "label_config": label_config_xml,
            },
            timeout=self.timeout,
        )
        create_resp.raise_for_status()
        return create_resp.json()

    def import_tasks_idempotent(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Import tasks into a Label Studio project while skipping existing sample_ids."""
        existing_resp = self.session.get(
            f"{self.base_url}/api/projects/{project_id}/tasks",
            timeout=self.timeout,
        )
        existing_ids: Set[str] = set()
        if existing_resp.status_code == 200:
            existing_tasks = existing_resp.json()
            if isinstance(existing_tasks, dict):
                existing_tasks = existing_tasks.get("tasks", [])
            for t in existing_tasks:
                sid = t.get("data", {}).get("sample_id")
                if sid:
                    existing_ids.add(sid)

        new_tasks = [
            t for t in tasks if t.get("data", {}).get("sample_id") not in existing_ids
        ]
        if not new_tasks:
            return {"imported": 0, "skipped_existing": len(tasks)}

        imp_resp = self.session.post(
            f"{self.base_url}/api/projects/{project_id}/import",
            json=new_tasks,
            timeout=self.timeout,
        )
        imp_resp.raise_for_status()
        return {"imported": len(new_tasks), "skipped_existing": len(tasks) - len(new_tasks)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Validate Label Studio XML config and generate/sync tasks."
    )
    parser.add_argument(
        "--xml-config",
        type=Path,
        default=base_dir / "configs" / "annotation_config.xml",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=base_dir / "configs" / "dataset_schema.json",
    )
    args = parser.parse_args()
    report = validate_annotation_config_xml(args.xml_config, args.schema)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
