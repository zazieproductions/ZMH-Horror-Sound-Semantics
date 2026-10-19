"""Beginner-friendly Hugging Face Hub discovery, streaming preview, and gated import helpers.

This module uses the Hub metadata API and datasets-server rows API. Preview requests are
limited to five rows and never clone a repository or download an entire dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

API_ROOT = "https://huggingface.co/api"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
PREFERRED_LICENSES = {"cc0-1.0", "cc-by-4.0"}
SEARCH_TERMS = ("horror sound effects", "sound events", "cinematic audio", "foley sound effects")
TIMEOUT = 20


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def discover_datasets(limit: int = 20) -> list[dict[str, Any]]:
    """Search the public Hub for relevant datasets, deduplicate, and rank CC0/CC-BY first."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    found: dict[str, dict[str, Any]] = {}
    for term in SEARCH_TERMS:
        url = f"{API_ROOT}/datasets"
        response = requests.get(url, params={"search": term, "limit": limit, "full": "true"}, timeout=TIMEOUT)
        response.raise_for_status()
        for item in response.json():
            dataset_id = item.get("id") or item.get("_id")
            if not dataset_id:
                continue
            details = _get_json(f"{API_ROOT}/datasets/{quote(dataset_id, safe='/')}")
            tags = details.get("tags", item.get("tags", [])) or []
            license_tag = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "unknown")
            card = details.get("cardData") or {}
            description = card.get("description") or details.get("description") or item.get("description") or "No dataset-card description found."
            splits = []
            try:
                split_data = _get_json(f"{DATASETS_SERVER}/splits", {"dataset": dataset_id})
                splits = [row.get("split", "") for row in split_data.get("splits", [])]
            except requests.RequestException:
                # Hub API discovery should still return candidates if datasets-server is unavailable.
                pass
            found[dataset_id] = {
                "dataset_id": dataset_id,
                "license": license_tag,
                "license_priority": "preferred" if license_tag.lower() in PREFERRED_LICENSES else "review_required",
                "size": _format_size(details.get("downloadsSize") or details.get("usedStorage")),
                "size_bytes": details.get("downloadsSize") or details.get("usedStorage"),
                "available_splits": splits,
                "description": str(description).strip(),
                "source_url": f"https://huggingface.co/datasets/{dataset_id}",
                "license_review": verify_license_metadata(details, license_tag, dataset_id),
            }
    return sorted(found.values(), key=lambda row: (row["license_priority"] != "preferred", row["dataset_id"]))[:limit]


def _format_size(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Not reported by Hub"
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1000 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1000
    return "Not reported by Hub"


def verify_license_metadata(details: dict[str, Any], license_tag: str, dataset_id: str | None = None) -> dict[str, Any]:
    """Screen Hub metadata and, when available, the public dataset README/card text.

    Automated text inspection is only a warning system—not legal verification.
    """
    card = details.get("cardData") or {}
    text = " ".join(str(card.get(key, "")) for key in ("license", "license_name", "description"))
    card_fetched = False
    if dataset_id:
        try:
            response = requests.get(f"https://huggingface.co/datasets/{quote(dataset_id, safe='/')}/raw/main/README.md", timeout=TIMEOUT, stream=True)
            if response.ok:
                body = next(response.iter_content(chunk_size=200_000), b"")[:200_000].decode("utf-8", errors="replace")
                text += " " + body
                card_fetched = bool(body)
            response.close()
        except requests.RequestException:
            pass
    issues = []
    if license_tag.lower() not in PREFERRED_LICENSES:
        issues.append("License tag is missing or not CC0/CC BY 4.0; do not import until rights are reviewed.")
    if not text.strip():
        issues.append("No usable license text found in Hub card metadata; inspect the repository README and source terms.")
    if re.search(r"non.?commercial|no.?derivatives|research.?only|all rights reserved", text, re.I):
        issues.append("Card text suggests restrictions; do not import without explicit rights-holder clarification.")
    if not issues:
        issues.append("Hub tag/card appears compatible, but this automated check is not legal verification. Confirm the source card, linked terms, and underlying asset rights.")
    return {"redistribution_rights_clear": False, "status": "manual_review_required", "notes": issues,
            "reviewed_card_metadata": bool(text.strip())}


def preview_dataset(dataset_id: str, split: str | None = None, count: int = 5) -> list[dict[str, Any]]:
    """Fetch up to five dataset-server rows without downloading the dataset repository."""
    if count != 5:
        raise ValueError("Preview is intentionally capped at exactly five examples.")
    params: dict[str, Any] = {"dataset": dataset_id, "offset": 0, "length": 5}
    if split:
        params["split"] = split
    result = _get_json(f"{DATASETS_SERVER}/rows", params)
    return result.get("rows", [])[:5]


def project_source_fields_to_zmh(row: dict[str, Any]) -> dict[str, Any]:
    """Map recognizable source columns to ZMH concepts without inventing labels.

    This is a partial annotation projection, not a valid original-audio training record:
    absent concepts remain absent and must be annotated by a human before schema export.
    """
    aliases = {
        "audio": ("audio", "audio_url", "file", "path"),
        "natural_language_caption": ("caption", "text", "description", "sentence"),
        "sound_event_category": ("label", "class", "category", "event"),
        "temporal_events": ("events", "segments", "timestamps"),
    }
    projection: dict[str, Any] = {}
    for target, candidates in aliases.items():
        for key in candidates:
            if key in row and row[key] not in (None, ""):
                projection[target] = row[key]
                break
    return projection


def _metadata_only(value: Any) -> Any:
    """Remove inline audio payloads while retaining safe references and descriptive metadata."""
    if isinstance(value, dict):
        return {key: _metadata_only(item) for key, item in value.items() if key.lower() not in {"bytes", "data", "array"}}
    if isinstance(value, list):
        return [_metadata_only(item) for item in value]
    return value


def _record_id(dataset_id: str, row: dict[str, Any]) -> str:
    row_id = row.get("id") if row.get("id") is not None else row.get("row_idx")
    if row_id is None:
        row_id = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{dataset_id}:{row_id}"


def import_approved_records(dataset_id: str, *, approve: bool = False, split: str | None = None,
                            limit: int = 5, output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Create a separate external-data manifest after explicit approval; audio is never fetched.

    Import is restricted to public CC0/CC BY 4.0 metadata and a small row cap. An approval
    flag acknowledges user review but does not substitute for legal review.
    """
    if not approve:
        raise PermissionError("Import requires --approve after you have reviewed the dataset card and asset rights.")
    if not 1 <= limit <= 100:
        raise ValueError("Import limit must be between 1 and 100; no bulk downloads are performed.")
    details = _get_json(f"{API_ROOT}/datasets/{quote(dataset_id, safe='/')}")
    tags = details.get("tags", []) or []
    license_tag = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), "unknown")
    review = verify_license_metadata(details, license_tag, dataset_id)
    if license_tag.lower() not in PREFERRED_LICENSES:
        raise PermissionError(f"Refusing import: license {license_tag!r} is not CC0 or CC BY 4.0.")
    params: dict[str, Any] = {"dataset": dataset_id, "offset": 0, "length": limit}
    if split:
        params["split"] = split
    rows = _get_json(f"{DATASETS_SERVER}/rows", params).get("rows", [])[:limit]
    # Retain source fields rather than inventing the ZMH original-audio labels.
    output = []
    revision = details.get("sha") or details.get("lastModified") or "unknown"
    for wrapped in rows:
        source_row = _metadata_only(wrapped.get("row", wrapped))
        output.append({
            "record_type": "third_party_huggingface_dataset",
            "source_dataset_id": dataset_id,
            "source_record_id": _record_id(dataset_id, {**source_row, "row_idx": wrapped.get("row_idx", source_row.get("id"))}),
            "source_revision": revision,
            "original_license": license_tag,
            "source_url": f"https://huggingface.co/datasets/{dataset_id}",
            "attribution_required": license_tag.lower() == "cc-by-4.0",
            "attribution_text": f"{dataset_id} — {license_tag}" if license_tag.lower() == "cc-by-4.0" else None,
            "license_review": review,
            "human_review_status": "unreviewed_provisional",
            "zmh_annotation_status": "not_annotated",
            "zmh_schema_mapping": project_source_fields_to_zmh(source_row),
            "source_row": source_row,
        })
    target = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "data" / "third_party" / "huggingface"
    target.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id)
    path = target / f"{safe_id}.jsonl"
    existing = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["source_record_id"]] = record
    for record in output:
        existing.setdefault(record["source_record_id"], record)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in existing.values()), encoding="utf-8")
    return list(existing.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover, preview, and carefully import Hugging Face audio datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Search Hub dataset cards (no audio download)")
    discover.add_argument("--limit", type=int, default=20)
    preview = sub.add_parser("preview", help="Stream five row examples from the Hub viewer")
    preview.add_argument("dataset_id")
    preview.add_argument("--split")
    imp = sub.add_parser("import", help="Save a small, approved metadata-only import to a separate directory")
    imp.add_argument("dataset_id")
    imp.add_argument("--split")
    imp.add_argument("--limit", type=int, default=5)
    imp.add_argument("--approve", action="store_true", help="Confirm you reviewed license and underlying asset rights")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            result = discover_datasets(args.limit)
        elif args.command == "preview":
            result = preview_dataset(args.dataset_id, args.split)
        else:
            result = import_approved_records(args.dataset_id, approve=args.approve, split=args.split, limit=args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except (requests.RequestException, ValueError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
