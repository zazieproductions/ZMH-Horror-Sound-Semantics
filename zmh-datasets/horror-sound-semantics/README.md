# ZMH Horror Sound Semantics (v0.1)

**Owner / Publisher:** Zazie Productions LLC — Zazie Media Holdings  
**Product Category:** Audio Machine Learning / Cinematic Sound Classification & Audio-Language Semantics  
**Dataset Version:** 0.1 (MVP) — 200 samples, 10 categories, 0 human-reviewed records  
**Parent repository:** Zazie fork of [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) (upstream platform: Apache-2.0)

> Start at the [repository root README](../../README.md) for the fork-level overview, install paths, and licensing summary. This file is the technical manual for the dataset itself.

---

## 1. Overview

**ZMH Horror Sound Semantics (v0.1)** is a modular, reproducible audio-semantics dataset and annotation pipeline built as an isolated extension inside Zazie Productions LLC's fork of Label Studio. It pairs a domain-specific horror/cinematic sound ontology with the labeling interface needed to audit it by ear.

It delivers:

- **200 original procedural horror audio samples** (`data/demo_audio/zmh_hss_v01_0001.wav` – `zmh_hss_v01_0200.wav`, 16.8 MB, 525 s total) synthesized across **10 cinematic horror sound-event categories** using deterministic digital signal processing (DSP) with zero third-party samples and zero pretrained generative AI weights.
- **12 factual acoustic signal measurements** per clip extracted via windowed FFT/STFT and temporal envelope analysis (`src/analyze_audio.py`).
- **38 structured semantic, technical, and provenance fields** per record (37 schema-required), enforced by a strict JSON Schema Draft 2020-12 specification (`configs/dataset_schema.json`).
- **200 unique natural-language captions** and **450 time-aligned event regions** available for waveform review.
- **A dedicated Label Studio audio-labeling configuration** (`configs/annotation_config.xml`, 17 interactive controls) and task/prediction integration (`src/label_studio_integration.py`) supporting waveform region tagging, controlled-vocabulary classification, natural-language caption refinement, and human review status tracking.
- **Strict IP and provenance separation** (`data/provenance_manifest.csv` and `docs/IP_OWNERSHIP.md`) distinguishing upstream Apache 2.0 software, Zazie-authored software/schemas, Zazie-controlled creative compilations, and noncopyrightable factual measurements / procedural waveforms.

### Measured properties of the committed data

| Property | Verified value |
|---|---|
| Records in `training_data.jsonl` | 200 (20 per category, 10 categories) |
| Rows × columns in `metadata.csv` | 100 × 31 |
| Tasks in `label_studio_tasks.json` | 100, each `data` + `predictions` |
| Audio format | 16,000 Hz, 16-bit PCM mono WAV |
| Total / min / max clip duration | 525.0 s / 2.00 s / 3.25 s |
| Audio payload on disk | 16,808,800 bytes (16.03 MiB) |
| Distinct `recipe_id` groups | 100 (10 recipes × 10 categories), 2 variants each |
| Splits | 120 train / 40 validation / 40 test |
| Recipes shared across splits | 0 |
| `temporal_events` segments | 199 |
| Distinct captions | 100 |
| `human_review_status` | `unreviewed_provisional` for all 200 |
| `generation_method` | `procedural_dsp_synthesis` for all 200 |
| Automated tests | 15 / 15 passing |

### Provisional `primary_emotion` distribution

`unease` 20 · `suspense` 17 · `anxiety` 13 · `shock` 12 · `disorientation` 11 · `dread` 10 · `anticipation` 10 · `uncertainty` 7
*(algorithmic heuristic output in v0.1 — not human listening judgments)*

---

## 2. Directory Structure

```text
zmh-datasets/horror-sound-semantics/
├── README.md                        # this file
├── requirements.txt                 # numpy, scipy, jsonschema, requests, pytest ($0 / CPU-only)
├── .gitignore                       # excludes bulk/high-res renders from Git
├── src/
│   ├── __init__.py
│   ├── generate_audio.py            # Deterministic 10-category procedural DSP audio synthesizer
│   ├── analyze_audio.py             # Factual spectral & temporal acoustic feature extractor
│   ├── generate_metadata.py         # Provisional label & candidate caption generator
│   ├── label_studio_integration.py  # Label Studio XML validator, task exporter & human review ingester
│   ├── validate_dataset.py          # Schema, checksum, audio readability & split-leakage validator
│   ├── export_dataset.py            # JSONL, CSV, provenance manifest & Label Studio JSON exporter
│   └── pipeline.py                  # Idempotent end-to-end orchestration pipeline
├── configs/
│   ├── annotation_config.xml        # Label Studio audio waveform + semantic labeling interface
│   └── dataset_schema.json          # JSON Schema (Draft 2020-12): 38 properties, controlled vocabularies
├── data/
│   ├── demo_audio/                  # 100 validated 16 kHz 16-bit PCM mono WAV files (8.4 MB)
│   ├── metadata.csv                 # Tabular dataset metadata (100 rows × 31 columns)
│   ├── training_data.jsonl          # Complete structured ML training records (100 JSON lines)
│   ├── provenance_manifest.csv      # Per-sample IP, provenance & licensing classification manifest
│   └── label_studio_tasks.json      # Pre-labeled Label Studio import tasks (`data` + `predictions`)
├── docs/
│   ├── DATASET_CARD.md              # Technical datasheet, distributions, splits & acoustic profiles
│   ├── ANNOTATION_GUIDELINES.md     # Human annotator handbook & controlled vocabulary definitions
│   ├── IP_OWNERSHIP.md              # 4-tier intellectual property & copyright boundary framework
│   ├── QUALITY_REPORT.md            # Empirical validation results, test audit & technical limitations
│   └── COMMERCIAL_LICENSE_DRAFT.md  # Draft commercial data license & commercialization strategy
└── tests/
    ├── test_generate_audio.py
    ├── test_analyze_audio.py
    ├── test_metadata_and_schema.py
    ├── test_label_studio_integration.py
    └── test_pipeline_and_exports.py
```

---

## 3. Quickstart & Reproducible Pipeline Execution

### Install Lightweight Dependencies ($0 Budget / CPU-Only)

```bash
cd zmh-datasets/horror-sound-semantics
pip install -r requirements.txt
```

If your environment is externally managed (PEP 668), use a virtualenv:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Run the Full End-to-End Pipeline (`Generate → Analyze → Propose Labels → Annotate → Validate → Export`)

```bash
python3 -m src.pipeline --count 100 --sample-rate 16000 --base-seed 202610
```

`--base-seed 202610` is the seed that reproduces the committed v0.1 waveforms byte-for-byte. Running the pipeline multiple times is **idempotent**: it verifies SHA-256 checksums, never duplicates records, and preserves any human-reviewed annotations (`human_reviewed_verified` / `human_reviewed_corrected`).

### Run Individual Pipeline Modules

```bash
# 1. Generate procedural audio only (sample rate: 16000 | 22050 | 44100 | 48000)
python3 -m src.generate_audio --output-dir data/demo_audio --count 100 --sample-rate 16000

# 2. Analyze a single WAV file or a directory of WAV files
python3 -m src.analyze_audio data/demo_audio/zmh_hss_v01_0001.wav
python3 -m src.analyze_audio data/demo_audio

# 3. Validate Label Studio XML configuration against JSON Schema vocabularies
python3 -m src.label_studio_integration \
    --xml-config configs/annotation_config.xml \
    --schema configs/dataset_schema.json

# 4. Run full dataset validation (schema, duplicates, WAV readability, SHA-256, split leakage)
python3 -m src.validate_dataset            # exits non-zero if anything fails

# 5. Export JSONL, CSV, provenance manifest, and Label Studio tasks
python3 -m src.export_dataset --input-jsonl data/training_data.jsonl --data-dir data
```

### Run Automated Test Suite

```bash
python3 -m pytest -v        # 15 tests: synthesis determinism, feature extraction,
                            # schema conformance, LS round-trip, idempotent pipeline
```

---

## Hugging Face Dataset Discovery and Import

For beginner-friendly, copy-paste examples of public Hub discovery, five-row streaming previews, license review, and metadata-only approved imports, see [`docs/HUGGINGFACE_IMPORT.md`](docs/HUGGINGFACE_IMPORT.md). The integration stores third-party records separately and never downloads audio payloads or imports outside the CC0/CC BY 4.0 allowlist.

## 4. Label Studio Integration Workflow

The labeling interface exposes **17 interactive controls**: 13 `<Choices>` groups (10 `sound_event_category`, 10 `acoustic_texture`, 8 `primary_emotion`, 8 `secondary_emotions`, 5 `intensity`, 5 `tension`, 7 `rhythmic_characteristics`, 7 `tonality`, 7 `perceived_spatial_properties`, 10 `cinematic_usage`, 4 `annotation_provenance`, 5 `human_review_status`, 4 `copyright_and_licensing_status`), 1 `<Labels>` group with 10 waveform region tags (`temporal_events`), 2 `<TextArea>` (`natural_language_caption`, `reviewer_notes`), and 1 `<Number>` (`annotation_confidence`). Audio playback binds to `$audio_url`.

1. **Run Label Studio** — `pip install . && label-studio start` from the repository root (or Docker; see the root README).
2. **Create a Project:**
   - Paste the contents of `configs/annotation_config.xml` into **Project Settings → Labeling Interface → Code**, or use `LabelStudioClient.create_or_get_project()` in `src/label_studio_integration.py`.
3. **Import Pre-Labeled Tasks:**
   - Import `data/label_studio_tasks.json` via the Label Studio UI or `LabelStudioClient.import_tasks_idempotent()`.
   - Each task includes `data` (sample metadata, synthesis parameters, measured acoustic features, and audio path) and `predictions` (`model_version: "zmh-hss-provisional-v0.1"`) pre-populating all 17 labeling controls.
   - Make `data/demo_audio/` reachable by the server (local storage directory or an HTTP media URL) so each task's `$audio_url` resolves.
4. **Human Review & Correction:**
   - Reviewers listen to the audio clip, inspect or adjust temporal regions and semantic tags, refine the `natural_language_caption` (removing the `[Provisional Algorithmic Caption]` prefix), and submit the annotation. Follow `docs/ANNOTATION_GUIDELINES.md`.
5. **Ingest Completed Reviews Back into the Dataset:**
   ```bash
   python3 -m src.pipeline --ingest-reviews path/to/label_studio_export.json
   ```
   - `ingest_human_annotations_from_label_studio()` automatically updates `human_review_status` (`human_reviewed_verified` or `human_reviewed_corrected`) and `annotation_provenance` (`human_expert_reviewed_v0.1` or `human_expert_corrected_v0.1`).
   - **Only records that have passed this step may be described as human-audited.** In v0.1 that count is 0.

---

## 5. Reading the Data Directly

```python
import json
from pathlib import Path

base = Path("zmh-datasets/horror-sound-semantics")
records = [json.loads(l) for l in open(base / "data/training_data.jsonl")]

rec = records[0]
x = rec["acoustic_features"]
print(rec["sample_id"], rec["sound_event_category"], rec["human_review_status"])
print(round(x["spectral_centroid_hz"], 1), x["crest_factor_db"], len(rec["temporal_events"]))
print(rec["synthesis_parameters"]["engine_name"], rec["synthesis_parameters"]["random_seed"])

train = [r for r in records if r["split"] == "train"]          # 60
audit = [r for r in records if r["human_review_status"] != "unreviewed_provisional"]  # 0 in v0.1
```

Filter on `human_review_status` before treating any affective label as a perceptual ground truth, and on `uses_pretrained_ai_model` / `uses_third_party_samples` (both `False` throughout) for provenance-clean subsets.

---

## 6. Known Limitations (stated up front)

1. **No human-reviewed annotations yet** — all 200 records are `unreviewed_provisional`; the review ingestion path is implemented and unit-tested but has not been run against real listening sessions.
2. **Procedural DSP, not acoustic Foley** — deterministic synthesis guarantees copyright cleanliness but omits the micro-acoustic complexity of physical sources (bowed waterphone, dry ice on sheet metal, real room impulse responses).
3. **16 kHz mono baseline** — keeps the committed payload at 8.4 MB and matches standard audio ML encoders (VGGish, Wav2Vec2, Whisper); use `--sample-rate 48000` for high-resolution renders, which are deliberately excluded from Git by `.gitignore`.
4. **v0.1 is small by design** — 100 clips across 10 classes is a schema and pipeline validation milestone, not a production-scale training corpus. See `docs/QUALITY_REPORT.md` for the full audit.

---

## 7. License & Intellectual Property Notice

- **Upstream Label Studio Platform:** All upstream Label Studio code in this repository remains licensed under the **Apache License, Version 2.0** (Copyright Heartex, Inc. / HumanSignal). See the root [`LICENSE`](../../LICENSE) and [`NOTICE`](../../NOTICE). Zazie asserts no ownership of upstream code, assets, or trademarks.
- **ZMH Dataset Tooling & Documentation:** Original scripts, schemas, and documentation inside `zmh-datasets/horror-sound-semantics/` are created for **Zazie Productions LLC — Zazie Media Holdings**. See [`docs/IP_OWNERSHIP.md`](docs/IP_OWNERSHIP.md) and [`docs/COMMERCIAL_LICENSE_DRAFT.md`](docs/COMMERCIAL_LICENSE_DRAFT.md) for detailed rights boundaries.
- **Deliberate caveat:** the 12 measured `acoustic_features` are noncopyrightable facts, and purely procedural waveforms do not automatically qualify for standalone sound-recording copyright under U.S. law. Rights claims are scoped to compilation, schema, documentation, and human-reviewed content accordingly.
