# ZMH Horror Sound Semantics (v0.1)

**Owner / Publisher:** Zazie Productions LLC — Zazie Media Holdings  
**Product Category:** Audio Machine Learning / Cinematic Sound Classification & Audio-Language Semantics  
**Dataset Version:** 0.1 (MVP)

---

## 1. Overview

**ZMH Horror Sound Semantics (v0.1)** is a modular, reproducible audio-semantics dataset and annotation pipeline built as an isolated extension within Zazie Productions LLC's fork of [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio).

It delivers:
- **100 original procedural horror audio samples** (`data/demo_audio/zmh_hss_v01_0001.wav` – `zmh_hss_v01_0100.wav`) synthesized across **10 cinematic horror sound-event categories** using deterministic digital signal processing (DSP) with zero third-party samples and zero pretrained generative AI weights.
- **12 factual acoustic signal measurements** per clip extracted via windowed FFT/STFT and temporal envelope analysis (`src/analyze_audio.py`).
- **19+ structured semantic and provenance fields** enforced by a strict JSON Schema Draft 2020-12 specification (`configs/dataset_schema.json`).
- **A dedicated Label Studio audio-labeling configuration** (`configs/annotation_config.xml`) and task/prediction integration (`src/label_studio_integration.py`) supporting waveform region tagging, controlled-vocabulary classification, natural-language caption refinement, and human review status tracking.
- **Strict IP and provenance separation** (`data/provenance_manifest.csv` and `docs/IP_OWNERSHIP.md`) distinguishing upstream Apache 2.0 software, Zazie-authored software/schemas, Zazie-controlled creative compilations, and noncopyrightable factual measurements / procedural waveforms.

---

## 2. Directory Structure

```text
zmh-datasets/horror-sound-semantics/
├── README.md
├── requirements.txt
├── .gitignore
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
│   └── dataset_schema.json          # Machine-readable JSON Schema (Draft 2020-12) & controlled vocabularies
├── data/
│   ├── demo_audio/                  # 100 validated 16 kHz 16-bit PCM mono WAV files (~12.7 MB total)
│   ├── metadata.csv                 # Tabular dataset metadata (100 rows x 31 columns)
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

### Run the Full End-to-End Pipeline (`Generate → Analyze → Propose Labels → Annotate → Validate → Export`)

```bash
python3 -m src.pipeline --count 100 --sample-rate 16000 --base-seed 202610
```

Running the pipeline multiple times is **idempotent**: it verifies SHA-256 checksums, never duplicates records, and preserves any human-reviewed annotations (`human_reviewed_verified` / `human_reviewed_corrected`).

### Run Individual Pipeline Modules

```bash
# 1. Generate procedural audio only
python3 -m src.generate_audio --output-dir data/demo_audio --count 100 --sample-rate 16000

# 2. Analyze a single WAV file or directory of WAV files
python3 -m src.analyze_audio data/demo_audio/zmh_hss_v01_0001.wav

# 3. Validate Label Studio XML configuration against JSON Schema vocabularies
python3 -m src.label_studio_integration --xml-config configs/annotation_config.xml --schema configs/dataset_schema.json

# 4. Run full dataset validation (schema, duplicates, WAV readability, SHA-256, split leakage)
python3 -m src.validate_dataset

# 5. Export JSONL, CSV, provenance manifest, and Label Studio tasks
python3 -m src.export_dataset --input-jsonl data/training_data.jsonl --data-dir data
```

### Run Automated Test Suite

```bash
python3 -m pytest -v
```

---

## 4. Label Studio Integration Workflow

1. **Create a Project in Label Studio:**
   - Paste the contents of `configs/annotation_config.xml` into **Project Settings → Labeling Interface → Code**, or use `LabelStudioClient.create_or_get_project()` in `src/label_studio_integration.py`.
2. **Import Pre-Labeled Tasks:**
   - Import `data/label_studio_tasks.json` via the Label Studio UI or `LabelStudioClient.import_tasks_idempotent()`.
   - Each task includes `data` (sample metadata, synthesis parameters, measured acoustic features, and audio path) and `predictions` (`model_version: "zmh-hss-provisional-v0.1"`) pre-populating all 16 labeling controls.
3. **Human Review & Correction:**
   - Reviewers listen to the audio clip, inspect or adjust temporal regions and semantic tags, refine the `natural_language_caption`, and submit the annotation.
4. **Ingest Completed Reviews Back into the Dataset:**
   ```bash
   python3 -m src.pipeline --ingest-reviews path/to/label_studio_export.json
   ```
   - `ingest_human_annotations_from_label_studio()` automatically updates `human_review_status` (`human_reviewed_verified` or `human_reviewed_corrected`) and `annotation_provenance` (`human_expert_reviewed_v0.1` or `human_expert_corrected_v0.1`).

---

## 5. License & Intellectual Property Notice

- **Upstream Label Studio Platform:** All upstream Label Studio code in this repository remains licensed under the **Apache License, Version 2.0** (Copyright Heartex, Inc. / HumanSignal). See root `LICENSE` and `NOTICE`.
- **ZMH Dataset Tooling & Documentation:** Original scripts and documentation inside `zmh-datasets/horror-sound-semantics/` are created for **Zazie Productions LLC — Zazie Media Holdings**. See `docs/IP_OWNERSHIP.md` and `docs/COMMERCIAL_LICENSE_DRAFT.md` for detailed rights boundaries.
