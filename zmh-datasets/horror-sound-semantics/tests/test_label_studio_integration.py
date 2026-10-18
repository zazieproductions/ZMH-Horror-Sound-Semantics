"""Tests for Label Studio XML configuration, task generation, and human review ingestion."""

import copy
import json
from pathlib import Path

from src.export_dataset import load_from_jsonl
from src.label_studio_integration import (
    ingest_human_annotations_from_label_studio,
    record_to_label_studio_task,
    validate_annotation_config_xml,
)
from src.validate_dataset import validate_records_against_schema

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "configs" / "annotation_config.xml"
SCHEMA_PATH = BASE_DIR / "configs" / "dataset_schema.json"
JSONL_PATH = BASE_DIR / "data" / "training_data.jsonl"


def test_annotation_config_xml_validates_against_schema() -> None:
    """configs/annotation_config.xml must pass structural and vocabulary alignment validation."""
    report = validate_annotation_config_xml(XML_PATH, SCHEMA_PATH)
    assert report["valid"] is True
    assert "sound_event_category" in report["choice_controls"]
    assert "primary_emotion" in report["choice_controls"]
    assert "human_review_status" in report["choice_controls"]
    assert "temporal_events" in report["label_controls"]


def test_record_to_label_studio_task_format() -> None:
    """Every dataset record must convert cleanly into Label Studio data + predictions format."""
    records = load_from_jsonl(JSONL_PATH)
    task = record_to_label_studio_task(records[0])
    assert "data" in task
    assert "predictions" in task
    assert task["data"]["sample_id"] == "zmh_hss_v01_0001"
    assert len(task["predictions"]) == 1

    pred = task["predictions"][0]
    assert pred["model_version"] == "zmh-hss-provisional-v0.1"
    from_names = {item["from_name"] for item in pred["result"]}
    assert "sound_event_category" in from_names
    assert "primary_emotion" in from_names
    assert "secondary_emotions" in from_names
    assert "natural_language_caption" in from_names
    assert "human_review_status" in from_names
    assert "temporal_events" in from_names


def test_human_review_and_correction_ingestion_workflow() -> None:
    """
    Simulated Label Studio completed annotations must update human_review_status and
    annotation_provenance accurately while preserving schema compliance.
    """
    records = load_from_jsonl(JSONL_PATH)[:3]

    # Task 1: Human reviewer verifies the provisional annotation without changing semantic fields
    task_verified = record_to_label_studio_task(records[0])
    task_verified["annotations"] = [
        {
            "id": 101,
            "completed_by": "auditor@zazieproductions.com",
            "created_at": "2026-10-18T15:00:00Z",
            "lead_time": 24.5,
            "was_cancelled": False,
            "result": copy.deepcopy(task_verified["predictions"][0]["result"]),
        }
    ]

    # Task 2: Human reviewer modifies primary_emotion and natural_language_caption
    task_corrected = record_to_label_studio_task(records[1])
    corrected_results = copy.deepcopy(task_corrected["predictions"][0]["result"])
    for item in corrected_results:
        if item["from_name"] == "primary_emotion":
            item["value"]["choices"] = ["shock"]
        elif item["from_name"] == "natural_language_caption":
            item["value"]["text"] = [
                "Human-audited horror drone featuring resonant tritone beating and sudden sub-bass drop."
            ]
    task_corrected["annotations"] = [
        {
            "id": 102,
            "completed_by": "senior_editor@zazieproductions.com",
            "created_at": "2026-10-18T15:05:00Z",
            "lead_time": 42.1,
            "was_cancelled": False,
            "result": corrected_results,
        }
    ]

    updated_records, stats = ingest_human_annotations_from_label_studio(
        ls_exported_tasks=[task_verified, task_corrected],
        existing_records=records,
    )

    assert len(updated_records) == 3
    assert stats["human_verified_count"] == 1
    assert stats["human_corrected_count"] == 1
    assert stats["unreviewed_count"] == 1

    assert updated_records[0]["human_review_status"] == "human_reviewed_verified"
    assert updated_records[0]["annotation_provenance"] == "human_expert_reviewed_v0.1"
    assert updated_records[0]["human_review_audit"]["was_corrected"] is False

    assert updated_records[1]["human_review_status"] == "human_reviewed_corrected"
    assert updated_records[1]["annotation_provenance"] == "human_expert_corrected_v0.1"
    assert updated_records[1]["primary_emotion"] == "shock"
    assert updated_records[1]["human_review_audit"]["was_corrected"] is True

    assert updated_records[2]["human_review_status"] == "unreviewed_provisional"

    # Confirm updated records still pass full JSON Schema validation
    schema_errors = validate_records_against_schema(updated_records, SCHEMA_PATH)
    assert schema_errors == []
