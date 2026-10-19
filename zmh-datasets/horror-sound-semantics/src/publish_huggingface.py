"""
Private Hugging Face Hub publishing + post-upload verification for ZMH Horror Sound Semantics (v0.1).

This module never changes the licence, never creates or flips a repository to public, and never
uploads anything outside the export folder built by :mod:`src.export_huggingface`.

Authentication: set ``HF_TOKEN`` in the environment (or run ``huggingface-cli login``). The token is
read by ``huggingface_hub`` and is never printed, logged, or written to disk by this module.

Usage::

    # 0. Build and locally verify the export (also run automatically before upload)
    python3 -m src.export_huggingface build

    # 1. Create the PRIVATE dataset repo (if needed) and upload the export folder
    python3 -m src.publish_huggingface upload --repo-id <namespace>/zmh-horror-sound-semantics

    # 2. Verify the Hub copy: file count, LFS SHA-256 vs local, metadata<->audio association,
    #    dataset viewer status, first-rows preview, and audio playback URL decoding
    python3 -m src.publish_huggingface verify --repo-id <namespace>/zmh-horror-sound-semantics
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .export_huggingface import (
    BASE_DIR,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    PUBLIC_REDISTRIBUTION_LICENSES,
    SPLITS,
    build_huggingface_export,
    parse_readme_front_matter,
    sha256_of_file,
    verify_huggingface_export,
)
from .export_dataset import load_from_jsonl

logger = logging.getLogger(__name__)

DATASETS_SERVER = "https://datasets-server.huggingface.co"
REPO_TYPE = "dataset"
IGNORE_PATTERNS = ("**/third_party/**", "**/.DS_Store", "**/__pycache__/**", "**/*.pyc")


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # pragma: no cover
        return None


def _require_private_placeholder_license(export_dir: Path) -> None:
    meta, _ = parse_readme_front_matter(export_dir / "README.md")
    license_value = str(meta.get("license", "")).lower()
    if license_value in PUBLIC_REDISTRIBUTION_LICENSES:
        raise PermissionError(
            f"README.md selects public redistribution licence '{license_value}'. "
            "Publishing under a public licence requires explicit approval from the rights holder; aborting."
        )


def _lfs_sha256(entry: Any) -> Optional[str]:
    """Return the LFS SHA-256 recorded by the Hub for a RepoFile entry (None for regular files)."""
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def _load_manifest(export_dir: Path) -> Dict[str, str]:
    listed: Dict[str, str] = {}
    for line in (export_dir / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
    return listed


# --------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------
def upload_private_dataset(
    repo_id: str,
    *,
    export_dir: Optional[Path] = None,
    base_dir: Path = BASE_DIR,
    rebuild: bool = True,
    skip_local_datasets_check: bool = False,
    dry_run: bool = False,
    api: Any = None,
) -> Dict[str, Any]:
    """Create (if needed) a PRIVATE dataset repo and upload the export folder.

    Aborts if the export fails local verification, if the README selects a public licence, or if the
    target repository already exists and is public.
    """
    export_dir = Path(export_dir) if export_dir else Path(base_dir) / "huggingface"

    if rebuild:
        build_huggingface_export(base_dir, export_dir)
    local = verify_huggingface_export(base_dir, export_dir, load_with_datasets=not skip_local_datasets_check)
    if not local["valid"]:
        raise ValueError("Local export verification failed; refusing to upload:\n  " + "\n  ".join(local["errors"]))
    _require_private_placeholder_license(export_dir)

    manifest = _load_manifest(export_dir)
    plan = {
        "repo_id": repo_id,
        "repo_type": REPO_TYPE,
        "private": True,
        "files": len(manifest) + 1,  # + MANIFEST.sha256 itself
        "wav_files": sum(1 for rel in manifest if rel.endswith(".wav")),
        "bytes": sum((export_dir / rel).stat().st_size for rel in manifest),
        "ignore_patterns": list(IGNORE_PATTERNS),
    }
    if dry_run:
        return {"dry_run": True, "plan": plan, "local_verification": local}

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=_token())

    url = api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
    if getattr(info, "private", None) is not True:
        raise PermissionError(
            f"Repository {repo_id} exists and is NOT private (private={getattr(info, 'private', None)}). "
            "This tool only publishes to private repositories; aborting without uploading."
        )

    commit = api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message=(
            "ZMH Horror Sound Semantics v0.1 — 200 original procedural clips, "
            "unreviewed algorithmic metadata (private evaluation upload)"
        ),
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    return {
        "dry_run": False,
        "plan": plan,
        "repo_url": str(url),
        "commit_url": getattr(commit, "commit_url", str(commit)),
        "private": True,
        "local_verification": {k: local[k] for k in ("valid", "total_rows", "wav_files", "counts_by_split")},
    }


# --------------------------------------------------------------------------------------
# Verify (Hub side)
# --------------------------------------------------------------------------------------
def verify_hub_dataset(
    repo_id: str,
    *,
    export_dir: Optional[Path] = None,
    base_dir: Path = BASE_DIR,
    check_viewer: bool = True,
    check_playback: bool = True,
    api: Any = None,
    session: Any = None,
) -> Dict[str, Any]:
    """Verify the uploaded repository against the local export and the Hub dataset viewer."""
    import requests

    export_dir = Path(export_dir) if export_dir else Path(base_dir) / "huggingface"
    errors: List[str] = []
    token = _token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    session = session or requests.Session()

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

    # 1. Repo must be private.
    info = api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
    private = getattr(info, "private", None)
    if private is not True:
        errors.append(f"repository is not private (private={private})")

    # 2. File inventory + LFS SHA-256 comparison with the local manifest.
    tree = list(api.list_repo_tree(repo_id=repo_id, repo_type=REPO_TYPE, recursive=True, expand=True))
    files = {t.path: t for t in tree if type(t).__name__ == "RepoFile"}
    hub_wavs = sorted(p for p in files if p.endswith(".wav"))
    per_split = {s: sum(1 for p in hub_wavs if p.startswith(f"data/{s}/")) for s in SPLITS}
    if len(hub_wavs) != EXPECTED_TOTAL:
        errors.append(f"Hub has {len(hub_wavs)} WAV files, expected {EXPECTED_TOTAL}")
    if per_split != EXPECTED_SPLIT_COUNTS:
        errors.append(f"Hub WAV files per split {per_split} != {EXPECTED_SPLIT_COUNTS}")
    if any("third_party" in p for p in files):
        errors.append("Hub repository contains third_party paths — not approved")

    local_manifest = _load_manifest(export_dir)
    missing_on_hub = sorted(set(local_manifest) - set(files))
    extra_on_hub = sorted(set(files) - set(local_manifest) - {"MANIFEST.sha256", ".gitattributes"})
    if missing_on_hub:
        errors.append(f"{len(missing_on_hub)} local files missing on Hub, e.g. {missing_on_hub[:5]}")
    if extra_on_hub:
        errors.append(f"unexpected files on Hub: {extra_on_hub[:10]}")

    lfs_checked = 0
    downloaded_checked = 0
    for rel, digest in local_manifest.items():
        entry = files.get(rel)
        if entry is None:
            continue
        hub_sha = _lfs_sha256(entry)
        if hub_sha:
            lfs_checked += 1
            if hub_sha != digest:
                errors.append(f"LFS SHA-256 mismatch on Hub for {rel}")
        elif not rel.endswith(".wav"):
            # Small regular files: download and hash.
            path = api.hf_hub_download(repo_id=repo_id, repo_type=REPO_TYPE, filename=rel)
            downloaded_checked += 1
            if sha256_of_file(Path(path)) != digest:
                errors.append(f"content SHA-256 mismatch on Hub for {rel}")
        else:
            errors.append(f"{rel} is a WAV but not stored via LFS on the Hub")

    # 3. Metadata integrity: Hub metadata.jsonl rows ↔ Hub WAV LFS hashes ↔ source records.
    source = {r["sample_id"]: r for r in load_from_jsonl(Path(base_dir) / "data" / "training_data.jsonl")}
    hub_rows_total = 0
    for split in SPLITS:
        rel = f"data/{split}/metadata.jsonl"
        if rel not in files:
            errors.append(f"missing {rel} on Hub")
            continue
        path = api.hf_hub_download(repo_id=repo_id, repo_type=REPO_TYPE, filename=rel)
        rows = load_from_jsonl(Path(path))
        hub_rows_total += len(rows)
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            errors.append(f"Hub {rel} has {len(rows)} rows, expected {EXPECTED_SPLIT_COUNTS[split]}")
        for row in rows:
            wav_rel = f"data/{split}/{row['file_name']}"
            entry = files.get(wav_rel)
            if entry is None:
                errors.append(f"{row['sample_id']}: metadata points at {wav_rel} which is not on the Hub")
                continue
            hub_sha = _lfs_sha256(entry)
            if hub_sha and hub_sha != row["sha256_checksum"]:
                errors.append(f"{row['sample_id']}: Hub WAV SHA-256 != row sha256_checksum")
            src = source.get(row["sample_id"])
            if src is None:
                errors.append(f"{row['sample_id']}: not in source training_data.jsonl")
            else:
                for key in (
                    "human_review_status",
                    "annotation_provenance",
                    "generation_method",
                    "uses_third_party_samples",
                ):
                    if row.get(key) != src[key]:
                        errors.append(f"{row['sample_id']}: {key} differs on Hub")

    report: Dict[str, Any] = {
        "repo_id": repo_id,
        "private": private,
        "hub_wav_files": len(hub_wavs),
        "hub_wav_files_per_split": per_split,
        "hub_metadata_rows": hub_rows_total,
        "lfs_sha256_checked": lfs_checked,
        "regular_files_hashed": downloaded_checked,
    }

    # 4. Dataset viewer (datasets-server). Private repos need PRO / Enterprise for the viewer.
    if check_viewer:
        report["viewer"] = _check_viewer(repo_id, headers, session, errors, check_playback=check_playback)

    # 5. Programmatic load (what `load_dataset(repo_id)` does), with audio decode of one example.
    report["datasets_library"] = _check_streaming_load(repo_id, token, errors)

    report["valid"] = not errors
    report["error_count"] = len(errors)
    report["errors"] = errors
    return report


def _check_viewer(
    repo_id: str, headers: Dict[str, str], session: Any, errors: List[str], *, check_playback: bool
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        r = session.get(f"{DATASETS_SERVER}/is-valid", params={"dataset": repo_id}, headers=headers, timeout=60)
        out["is_valid_http"] = r.status_code
        out["is_valid"] = (
            r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300]
        )
        if r.status_code != 200 or not (isinstance(out["is_valid"], dict) and out["is_valid"].get("viewer")):
            errors.append(
                "dataset viewer not (yet) available: "
                f"HTTP {r.status_code} {out['is_valid']}. Note: the viewer is only enabled on PRIVATE datasets for "
                "PRO users / Enterprise Hub organisations, and it can take a few minutes after upload."
            )
            return out
        r = session.get(f"{DATASETS_SERVER}/splits", params={"dataset": repo_id}, headers=headers, timeout=60)
        splits = [s["split"] for s in r.json().get("splits", [])] if r.ok else []
        out["splits"] = splits
        if sorted(splits) != sorted(SPLITS):
            errors.append(f"viewer splits {splits} != {list(SPLITS)}")
        r = session.get(
            f"{DATASETS_SERVER}/rows",
            params={"dataset": repo_id, "config": "default", "split": "train", "offset": 0, "length": 5},
            headers=headers,
            timeout=60,
        )
        if not r.ok:
            errors.append(f"viewer /rows failed: HTTP {r.status_code} {r.text[:200]}")
            return out
        rows = r.json().get("rows", [])
        out["preview_rows"] = len(rows)
        out["preview_columns"] = len(r.json().get("features", []))
        if len(rows) != 5:
            errors.append(f"viewer returned {len(rows)} preview rows, expected 5")
        if rows:
            first = rows[0]["row"]
            out["first_row_sample_id"] = first.get("sample_id")
            out["first_row_human_review_status"] = first.get("human_review_status")
            audio_cell = first.get("audio")
            src_url = None
            if isinstance(audio_cell, list) and audio_cell and isinstance(audio_cell[0], dict):
                src_url = audio_cell[0].get("src")
            elif isinstance(audio_cell, dict):
                src_url = audio_cell.get("src")
            out["audio_src_present"] = bool(src_url)
            if not src_url:
                errors.append("viewer preview has no playable audio src for the first row")
            elif check_playback:
                out["playback"] = _check_playback_url(src_url, headers, session, errors)
    except Exception as exc:  # pragma: no cover - network dependent
        errors.append(f"viewer check failed: {type(exc).__name__}: {exc}")
    return out


def _check_playback_url(src_url: str, headers: Dict[str, str], session: Any, errors: List[str]) -> Dict[str, Any]:
    """Download the viewer's audio asset and decode it — the same bytes the browser player streams."""
    out: Dict[str, Any] = {"url_host": src_url.split("/")[2] if "//" in src_url else None}
    r = session.get(src_url, headers=headers, timeout=60)
    out["http"] = r.status_code
    out["content_type"] = r.headers.get("content-type")
    out["bytes"] = len(r.content)
    if not r.ok or not r.content:
        errors.append(f"audio asset not downloadable: HTTP {r.status_code}")
        return out
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(r.content))
        out.update({"decoded": True, "sample_rate": int(sr), "num_samples": int(len(data))})
        if len(data) == 0:
            errors.append("audio asset decoded to 0 samples")
    except Exception as exc:
        out.update({"decoded": False, "decode_error": f"{type(exc).__name__}: {exc}"})
        errors.append(f"audio asset could not be decoded locally: {exc}")
    return out


def _check_streaming_load(repo_id: str, token: Optional[str], errors: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"attempted": False}
    try:
        import datasets
    except ImportError:
        out["reason"] = "datasets not installed"
        return out
    out["attempted"] = True
    try:
        ds = datasets.load_dataset(repo_id, split="test", streaming=True, token=token)
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
        example = next(iter(ds))
        raw = example["audio"]
        import soundfile as sf

        buf = raw["bytes"] if raw.get("bytes") else None
        if buf is None and raw.get("path"):
            import requests

            buf = requests.get(
                raw["path"], headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60
            ).content
        data, sr = sf.read(io.BytesIO(buf))
        out.update(
            {
                "loaded": True,
                "sample_id": example.get("sample_id"),
                "human_review_status": example.get("human_review_status"),
                "sample_rate": int(sr),
                "num_samples": int(len(data)),
                "sha256_matches_row": hashlib.sha256(buf).hexdigest() == example.get("sha256_checksum"),
            }
        )
        if not out["sha256_matches_row"]:
            errors.append("streamed audio bytes do not match the row's sha256_checksum")
    except Exception as exc:  # pragma: no cover - network dependent
        out.update({"loaded": False, "error": f"{type(exc).__name__}: {exc}"})
        errors.append(f"datasets.load_dataset({repo_id!r}) failed: {exc}")
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Private Hugging Face upload + verification for the ZMH export.")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="create the PRIVATE repo if needed and upload the export folder")
    up.add_argument("--repo-id", required=True, help="e.g. <namespace>/zmh-horror-sound-semantics")
    up.add_argument("--export-dir", type=Path, default=None)
    up.add_argument("--base-dir", type=Path, default=BASE_DIR)
    up.add_argument("--no-rebuild", action="store_true", help="upload the existing export without rebuilding")
    up.add_argument("--skip-local-datasets-check", action="store_true")
    up.add_argument("--dry-run", action="store_true", help="verify locally and print the upload plan only")

    ver = sub.add_parser("verify", help="verify the uploaded repository and the dataset viewer")
    ver.add_argument("--repo-id", required=True)
    ver.add_argument("--export-dir", type=Path, default=None)
    ver.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ver.add_argument("--no-viewer", action="store_true", help="skip datasets-server viewer checks")
    ver.add_argument("--no-playback", action="store_true", help="skip downloading/decoding the preview audio asset")

    args = parser.parse_args(argv)
    if args.command == "upload":
        report = upload_private_dataset(
            args.repo_id,
            export_dir=args.export_dir,
            base_dir=args.base_dir,
            rebuild=not args.no_rebuild,
            skip_local_datasets_check=args.skip_local_datasets_check,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    report = verify_hub_dataset(
        args.repo_id,
        export_dir=args.export_dir,
        base_dir=args.base_dir,
        check_viewer=not args.no_viewer,
        check_playback=not args.no_playback,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
