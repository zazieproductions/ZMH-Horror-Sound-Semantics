"""Offline tests for the Hugging Face AudioFolder export and the private-publish guard rails."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import export_huggingface as hfx
from src import publish_huggingface as pub
from src.export_dataset import load_from_jsonl

BASE_DIR = Path(__file__).resolve().parent.parent
pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def built_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the export once into a temp dir (README/LICENSE copied from the tracked card)."""
    out = tmp_path_factory.mktemp("hf_export")
    shutil.copyfile(BASE_DIR / "huggingface" / "README.md", out / "README.md")
    shutil.copyfile(BASE_DIR / "huggingface" / "LICENSE.md", out / "LICENSE.md")
    report = hfx.build_huggingface_export(BASE_DIR, out)
    assert report["records_exported"] == 200
    return out


def test_export_contains_all_200_clips_with_correct_split_layout(built_export: Path) -> None:
    for split, expected in hfx.EXPECTED_SPLIT_COUNTS.items():
        split_dir = built_export / "data" / split
        wavs = sorted(p.name for p in split_dir.glob("*.wav"))
        rows = load_from_jsonl(split_dir / "metadata.jsonl")
        assert len(wavs) == len(rows) == expected
        assert wavs == sorted(r["file_name"] for r in rows)
        assert all(r["split"] == split for r in rows)
    assert sum(1 for _ in (built_export / "data").rglob("*.wav")) == 200
    assert not list(built_export.rglob("third_party"))


def test_export_preserves_provenance_and_review_status_verbatim(built_export: Path) -> None:
    source = {r["sample_id"]: r for r in load_from_jsonl(BASE_DIR / "data" / "training_data.jsonl")}
    with open(BASE_DIR / "data" / "provenance_manifest.csv", encoding="utf-8", newline="") as f:
        manifest = {row["sample_id"]: row for row in csv.DictReader(f)}
    rows = [r for s in hfx.SPLITS for r in load_from_jsonl(built_export / "data" / s / "metadata.jsonl")]
    assert len(rows) == 200
    for row in rows:
        src = source[row["sample_id"]]
        assert row["human_review_status"] == src["human_review_status"] == "unreviewed_provisional"
        assert (
            row["annotation_provenance"]
            == src["annotation_provenance"]
            == "algorithmic_dsp_and_acoustic_heuristic_v0.1"
        )
        assert row["generation_method"] == "procedural_dsp_synthesis"
        assert row["uses_third_party_samples"] is False and row["uses_pretrained_ai_model"] is False
        assert row["content_origin"] == hfx.CONTENT_ORIGIN_ZMH_PROCEDURAL
        assert row["sha256_checksum"] == src["sha256_checksum"] == manifest[row["sample_id"]]["sha256_checksum"]
        assert row["natural_language_caption"] == src["natural_language_caption"]
        for col in hfx.PROVENANCE_JOIN_COLUMNS:
            assert row[col] == manifest[row["sample_id"]][col]
        assert (
            json.loads(row["synthesis_parameters"]["additional_params_json"])
            == src["synthesis_parameters"]["additional_params"]
        )
        assert row["temporal_events"] == src["temporal_events"]
    assert sum(len(r["temporal_events"]) for r in rows) == 450


def test_local_verification_passes_without_datasets_library(built_export: Path) -> None:
    report = hfx.verify_huggingface_export(BASE_DIR, built_export, load_with_datasets=False)
    assert report["errors"] == []
    assert report["valid"] is True
    assert report["wav_files"] == 200 and report["total_rows"] == 200
    assert report["human_review_status_breakdown"] == {"unreviewed_provisional": 200}
    assert report["third_party_records"] == 0


def test_verification_detects_tampered_audio_and_missing_rows(built_export: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(built_export, broken)
    wav = next((broken / "data" / "test").glob("*.wav"))
    wav.write_bytes(wav.read_bytes()[:-16] + b"\x00" * 16)
    meta = broken / "data" / "validation" / "metadata.jsonl"
    lines = meta.read_text(encoding="utf-8").splitlines()
    meta.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    report = hfx.verify_huggingface_export(BASE_DIR, broken, load_with_datasets=False)
    assert report["valid"] is False
    assert any("SHA-256 mismatch" in e for e in report["errors"])
    assert any("WAV files and metadata rows differ" in e for e in report["errors"])


def test_build_refuses_third_party_or_non_procedural_records(tmp_path: Path) -> None:
    """The governance gate runs before any audio is copied, so a metadata-only base dir suffices."""
    base = tmp_path / "base"
    (base / "data").mkdir(parents=True)
    records = load_from_jsonl(BASE_DIR / "data" / "training_data.jsonl")
    records[3]["uses_third_party_samples"] = True
    with open(base / "data" / "training_data.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    shutil.copyfile(BASE_DIR / "data" / "provenance_manifest.csv", base / "data" / "provenance_manifest.csv")
    with pytest.raises(ValueError, match="Third-party content requires separate approval"):
        hfx.build_huggingface_export(base, tmp_path / "out")
    assert not (tmp_path / "out" / "data").exists()


def test_card_wording_guard_blocks_affirmative_expert_validation_claims() -> None:
    assert hfx._check_card_wording("These labels are expert-validated by sound designers.")
    assert hfx._check_card_wording("All captions were human-verified.")
    assert hfx._check_card_wording("not expert-validated, not human-verified, awaiting review") == []
    assert hfx._check_card_wording("There are no expert-validated records in v0.1.") == []


def test_tracked_card_front_matter_is_private_placeholder_and_matches_counts() -> None:
    meta, body = hfx.parse_readme_front_matter(BASE_DIR / "huggingface" / "README.md")
    assert meta["license"] == "other"
    assert meta["license"] not in hfx.PUBLIC_REDISTRIBUTION_LICENSES
    assert meta["license_name"]
    splits = {s["name"]: s["num_examples"] for s in meta["dataset_info"]["splits"]}
    assert splits == hfx.EXPECTED_SPLIT_COUNTS
    assert {df["split"] for df in meta["configs"][0]["data_files"]} == set(hfx.SPLITS)
    feature_names = [f["name"] for f in meta["dataset_info"]["features"]]
    assert feature_names[0] == "audio" and "file_name" not in feature_names
    assert hfx.verify_readme(BASE_DIR / "huggingface", hfx.EXPECTED_SPLIT_COUNTS) == []
    assert "unreviewed_provisional" in body and "expert-validated" in body  # negated usage only
    assert hfx._check_card_wording(body) == []


def test_verify_readme_rejects_public_license_and_count_drift(built_export: Path, tmp_path: Path) -> None:
    out = tmp_path / "card"
    out.mkdir()
    text = (built_export / "README.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(text.replace("license: other", "license: cc-by-4.0", 1), encoding="utf-8")
    errors = hfx.verify_readme(out, hfx.EXPECTED_SPLIT_COUNTS)
    assert any("public redistribution licence" in e for e in errors)
    (out / "README.md").write_text(text, encoding="utf-8")
    errors = hfx.verify_readme(out, {"train": 119, "validation": 40, "test": 40})
    assert any("num_examples" in e for e in errors)


def test_feature_yaml_dialects_normalise_equal() -> None:
    legacy = [{"name": "x", "sequence": "string"}, {"name": "y", "list": [{"name": "a", "dtype": "float64"}]}]
    modern = [{"name": "x", "list": "string"}, {"name": "y", "list": [{"name": "a", "dtype": "float64"}]}]
    assert hfx.normalize_features_yaml(legacy) == hfx.normalize_features_yaml(modern)


def test_upload_dry_run_plans_private_repo_without_network(built_export: Path) -> None:
    report = pub.upload_private_dataset(
        "zazie/zmh-test",
        export_dir=built_export,
        base_dir=BASE_DIR,
        rebuild=False,
        skip_local_datasets_check=True,
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["plan"]["private"] is True
    assert report["plan"]["wav_files"] == 200
    assert report["local_verification"]["valid"] is True


def test_upload_aborts_if_existing_repo_is_public(built_export: Path) -> None:
    api = Mock()
    api.repo_info.return_value = Mock(private=False)
    with pytest.raises(PermissionError, match="NOT private"):
        pub.upload_private_dataset(
            "zazie/zmh-test",
            export_dir=built_export,
            base_dir=BASE_DIR,
            rebuild=False,
            skip_local_datasets_check=True,
            api=api,
        )
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is True
    api.upload_folder.assert_not_called()


def test_upload_proceeds_only_for_private_repo_and_never_uploads_third_party(built_export: Path) -> None:
    api = Mock()
    api.repo_info.return_value = Mock(private=True)
    api.create_repo.return_value = "https://huggingface.co/datasets/zazie/zmh-test"
    api.upload_folder.return_value = Mock(commit_url="https://huggingface.co/datasets/zazie/zmh-test/commit/abc")
    report = pub.upload_private_dataset(
        "zazie/zmh-test",
        export_dir=built_export,
        base_dir=BASE_DIR,
        rebuild=False,
        skip_local_datasets_check=True,
        api=api,
    )
    assert report["private"] is True
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_type"] == "dataset"
    assert "**/third_party/**" in kwargs["ignore_patterns"]
    assert Path(kwargs["folder_path"]) == built_export


def test_upload_refuses_public_license_in_card(built_export: Path, tmp_path: Path) -> None:
    out = tmp_path / "pub_card"
    shutil.copytree(built_export, out)
    readme = out / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("license: other", "license: cc0-1.0", 1), encoding="utf-8"
    )
    with pytest.raises((PermissionError, ValueError), match="public redistribution licence"):
        pub.upload_private_dataset(
            "zazie/zmh-test",
            export_dir=out,
            base_dir=BASE_DIR,
            rebuild=False,
            skip_local_datasets_check=True,
            api=Mock(),
        )


class _FakeRepoFile:  # mimics huggingface_hub.hf_api.RepoFile (type name matters to the verifier)
    def __init__(self, path: str, lfs_sha: str | None) -> None:
        self.path = path
        self.lfs = Mock(sha256=lfs_sha) if lfs_sha else None


RepoFile = type("RepoFile", (_FakeRepoFile,), {})


class _FakeResponse:
    def __init__(
        self, status: int = 200, payload=None, content: bytes = b"", content_type: str = "application/json"
    ) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.content = content
        self.headers = {"content-type": content_type}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


def _fake_hub(built_export: Path, *, drop_wav: bool = False, viewer_ok: bool = True):
    """Build a fake HfApi + requests.Session pair that mirrors the local export."""
    manifest = {}
    for line in (built_export / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        manifest[rel] = digest
    tree = [RepoFile(rel, digest if rel.endswith(".wav") else None) for rel, digest in manifest.items()]
    tree += [RepoFile("MANIFEST.sha256", None), RepoFile(".gitattributes", None)]
    if drop_wav:
        tree = [t for t in tree if t.path != "data/test/zmh_hss_v01_0200.wav"]
    api = Mock()
    api.repo_info.return_value = Mock(private=True)
    api.list_repo_tree.return_value = tree
    api.hf_hub_download.side_effect = lambda repo_id, repo_type, filename: str(built_export / filename)

    first_wav = next((built_export / "data" / "train").glob("*.wav"))
    rows_payload = {
        "features": [{"name": "audio"}] * 46,
        "rows": [
            {
                "row_idx": i,
                "row": {
                    "sample_id": f"zmh_hss_v01_{i + 1:04d}",
                    "human_review_status": "unreviewed_provisional",
                    "audio": [
                        {"src": "https://datasets-server.huggingface.co/assets/x/audio.wav", "type": "audio/wav"}
                    ],
                },
            }
            for i in range(5)
        ],
    }

    def get(url, params=None, headers=None, timeout=None):
        if url.endswith("/is-valid"):
            return _FakeResponse(
                200 if viewer_ok else 404,
                {"viewer": True, "preview": True} if viewer_ok else {"error": "not available"},
            )
        if url.endswith("/splits"):
            return _FakeResponse(200, {"splits": [{"split": s} for s in ("train", "validation", "test")]})
        if url.endswith("/rows"):
            return _FakeResponse(200, rows_payload)
        if url.endswith("audio.wav"):
            return _FakeResponse(200, None, content=first_wav.read_bytes(), content_type="audio/wav")
        raise AssertionError(f"unexpected URL {url}")

    session = Mock()
    session.get.side_effect = get
    return api, session


def test_hub_verification_passes_against_faithful_mirror(built_export: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("soundfile")
    monkeypatch.setattr(
        pub, "_check_streaming_load", lambda repo_id, token, errors: {"attempted": False, "reason": "mocked"}
    )
    api, session = _fake_hub(built_export)
    report = pub.verify_hub_dataset(
        "zazie/zmh-test", export_dir=built_export, base_dir=BASE_DIR, api=api, session=session
    )
    assert report["errors"] == []
    assert report["valid"] is True
    assert report["hub_wav_files"] == 200
    assert report["hub_wav_files_per_split"] == hfx.EXPECTED_SPLIT_COUNTS
    assert report["hub_metadata_rows"] == 200
    assert report["lfs_sha256_checked"] == 200
    assert report["viewer"]["preview_rows"] == 5
    assert report["viewer"]["playback"]["decoded"] is True
    assert report["viewer"]["playback"]["sample_rate"] == 16000


def test_hub_verification_flags_missing_audio_and_unavailable_viewer(
    built_export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pub, "_check_streaming_load", lambda repo_id, token, errors: {"attempted": False})
    api, session = _fake_hub(built_export, drop_wav=True, viewer_ok=False)
    api.repo_info.return_value = Mock(private=False)
    report = pub.verify_hub_dataset(
        "zazie/zmh-test", export_dir=built_export, base_dir=BASE_DIR, api=api, session=session
    )
    assert report["valid"] is False
    joined = "\n".join(report["errors"])
    assert "not private" in joined
    assert "199 WAV files" in joined
    assert "zmh_hss_v01_0200" in joined
    assert "PRO users / Enterprise" in joined
