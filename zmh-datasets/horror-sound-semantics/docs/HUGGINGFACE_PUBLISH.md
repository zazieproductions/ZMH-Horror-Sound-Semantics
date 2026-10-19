# Publishing the dataset to a private Hugging Face Hub repository

This guide covers the **outbound** direction: packaging the 200 committed clips and their metadata as a
Hub-compatible dataset, uploading it to a **private** repository, and verifying the upload. (The inbound
direction — discovering and importing third-party Hub datasets — is documented in
[`HUGGINGFACE_IMPORT.md`](HUGGINGFACE_IMPORT.md) and is deliberately kept separate.)

## What gets published

`python3 -m src.export_huggingface build` assembles `huggingface/` in the Hub **AudioFolder** layout:

```text
huggingface/
├── README.md                     # Hub dataset card: YAML front matter + card body (tracked, hand-maintained)
├── LICENSE.md                    # licensing status notice — no public licence selected (tracked)
├── .gitattributes                # generated: LFS rules
├── MANIFEST.sha256               # generated: SHA-256 of every file
├── data/
│   ├── train/       metadata.jsonl (tracked) + 120 WAV (copied from data/demo_audio, SHA-256 verified)
│   ├── validation/  metadata.jsonl (tracked) +  40 WAV
│   └── test/        metadata.jsonl (tracked) +  40 WAV
└── supplementary/                # generated copies: provenance manifest, JSON Schema, LS config, IP framework, guidelines
```

Each `metadata.jsonl` row is a one-to-one projection of the matching `data/training_data.jsonl` record
(`audio_filename` → `file_name`, nested `additional_params` → `additional_params_json`) **plus** the per-sample
IP columns from `data/provenance_manifest.csv` and an explicit `content_origin` marker. Nothing is renamed,
re-scored, or re-labelled: `human_review_status`, `annotation_provenance`, `copyright_and_licensing_status`,
`generation_method`, the three provenance booleans and every SHA-256 are carried over verbatim.

### Governance rules enforced by the tooling

| Rule | Where enforced |
|---|---|
| Only original ZMH procedural audio is exported (`generation_method == procedural_dsp_synthesis`, all provenance flags `false`). Anything else, including `data/third_party/`, aborts the build. | `export_huggingface.build_huggingface_export` |
| The card may not describe unreviewed labels as expert-/human-validated (affirmative phrasing is rejected; negated phrasing is allowed). | `export_huggingface.verify_readme` |
| The card's `license` must not be a public redistribution licence unless the rights holder approves one; the current value is the placeholder `other` / `zmh-private-evaluation-no-license-selected`. | `verify_readme` and `publish_huggingface.upload_private_dataset` |
| Repositories are created `private=True`; an existing **public** repository aborts the upload. Nothing in the tooling can make a repo public. | `publish_huggingface.upload_private_dataset` |
| Split counts, feature list, SHA-256s and the README's `dataset_info` must agree with the files on disk before upload. | `verify_huggingface_export` |

## Step-by-step

```bash
cd zmh-datasets/horror-sound-semantics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-hub.txt            # huggingface_hub, datasets, soundfile, librosa (+ core deps)

# 1. Build + verify locally (loads the folder with `datasets`, decodes one clip per split)
python3 -m src.export_huggingface build

# 2. Authenticate — never paste the token into files or chat
export HF_TOKEN=hf_...            # or: huggingface-cli login

# 3. Dry run: shows the upload plan, uploads nothing
python3 -m src.publish_huggingface upload --repo-id <namespace>/zmh-horror-sound-semantics --dry-run

# 4. Upload to a PRIVATE dataset repository (created if missing)
python3 -m src.publish_huggingface upload --repo-id <namespace>/zmh-horror-sound-semantics

# 5. Verify the Hub copy
python3 -m src.publish_huggingface verify --repo-id <namespace>/zmh-horror-sound-semantics
```

`verify` checks, in order:

1. the repository is private;
2. the Hub file tree contains exactly 200 WAV files (120 / 40 / 40 per split) and no unexpected or `third_party` paths;
3. every WAV's **Hub-side LFS SHA-256** equals the local `MANIFEST.sha256` entry *and* the `sha256_checksum` in the
   row that references it (metadata ↔ audio association on the Hub itself); regular files are downloaded and hashed;
4. the three `metadata.jsonl` files have the right row counts and their review-status / provenance fields match
   the source records;
5. the dataset viewer (`datasets-server`): `is-valid`, the three splits, a 5-row preview of `train`, and the
   preview's audio asset is downloaded and decoded with `soundfile` (the same bytes the browser player streams);
6. `datasets.load_dataset(repo_id, split="test", streaming=True)` returns an example whose audio bytes hash to the
   row's `sha256_checksum`.

The command exits non-zero and lists every failed check.

### Notes on the viewer for private repositories

The Hub only renders the dataset viewer for **private** datasets when the owner is a PRO user or the repository
belongs to an Enterprise Hub organisation. On a free account the files, card, and `load_dataset` all work, but
`verify` will report `dataset viewer not (yet) available` for step 5. The viewer can also take a few minutes to
build after the first upload; rerun `verify` rather than re-uploading.

## Decisions that require explicit approval from the rights holder

The tooling will refuse to proceed, or the card will fail verification, until each of these is decided
deliberately and edited by hand:

1. **Public redistribution licence** — change `license:` / `license_name:` / `license_link:` in
   `huggingface/README.md` and rewrite `huggingface/LICENSE.md`. Until then the placeholder stays.
2. **Visibility** — the tooling never flips a repository to public; do that in the Hub UI only after (1).
3. **Third-party content** — approved imports must ship as their own config with their own licence and attribution
   columns; they are never merged into the `default` config. Extend `export_huggingface.py` deliberately rather
   than relaxing `is_exportable_zmh_original`.
4. **Human review claims** — only after Label Studio reviews are ingested and `human_review_status` changes should
   the card's wording about validation change.

## Keeping the card and the data in sync

* After regenerating audio or metadata, rerun `python3 -m src.export_huggingface build`; the per-split
  `metadata.jsonl` files are tracked so the diff shows exactly what changed.
* If the feature set changes, regenerate the `dataset_info` block with
  `python3 -m src.export_huggingface dataset-info` and paste it into the README front matter —
  `verify` fails on drift.
* `tests/test_huggingface_export.py` covers the export, the guard rails, and a mocked Hub-side verification.
