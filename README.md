# ZMH Horror Sound Semantics

**Provenance-audited cinematic horror audio dataset + Label Studio annotation pipeline** — built as an isolated extension inside Zazie Productions LLC's fork of [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio).

![Samples](https://img.shields.io/badge/samples-100-informational) ![Upstream](https://img.shields.io/badge/Label%20Studio-Apache%202.0-blue) ![Dataset](https://img.shields.io/badge/version-0.1_MVP-orange) ![Tests](https://img.shields.io/badge/pytest-15%2F15_passing-brightgreen) ![Review](https://img.shields.io/badge/human--reviewed-0%20of%20100-yellow)

| | |
|---|---|
| **Owner / Publisher** | Zazie Productions LLC — Zazie Media Holdings |
| **Dataset version** | 0.1 (MVP) |
| **Domain** | Audio ML · cinematic sound classification · audio-language semantics · affective sound computing |
| **Dataset home** | [`zmh-datasets/horror-sound-semantics/`](zmh-datasets/horror-sound-semantics/) |

---

## At a glance

Every figure below is measured from the files committed in this repository and is reproduced by `python3 -m src.validate_dataset`.

| Property | Value |
|---|---|
| Audio samples | **100** validated clips (`zmh_hss_v01_0001` – `zmh_hss_v01_0100`) |
| Sound-event categories | **10**, perfectly balanced (10 clips each) |
| Audio format | 16,000 Hz · 16-bit PCM · mono WAV |
| Total audio duration | **261.5 s** (4 min 21.5 s); clips run 2.00 s – 3.25 s |
| Committed audio payload | **8,372,400 bytes** (≈ 8.4 MB, 7.98 MiB) |
| Schema fields per record | **38** top-level properties, **37** required (JSON Schema Draft 2020-12) |
| Acoustic measurements per clip | **12** spectral & temporal features (windowed FFT/STFT + envelope analysis) |
| Natural-language captions | **100** unique, caption-per-clip |
| Time-aligned event regions | **199** `temporal_events` segments |
| Labeling interface | **17** interactive controls in `configs/annotation_config.xml` |
| Splits | 60 `train` / 20 `validation` / 20 `test` |
| Cross-split recipe leakage | **0** (30 / 10 / 10 mutually disjoint `recipe_id` groups) |
| Third-party samples or pretrained AI weights | **0** — 100% deterministic DSP synthesis |
| Human-reviewed annotations | **0** — all records honestly flagged `unreviewed_provisional` |
| Automated test suite | **15 / 15 passing** (`python3 -m pytest`) |

### The 10 categories and their synthesis engines

| Category | `n` | DSP engine | Mean spectral centroid | Mean low-freq ratio (<250 Hz) |
|---|---:|---|---:|---:|
| `dissonant_drones` | 10 | `detuned_tritone_cluster_drone_v1` | 143.0 Hz | 0.9108 |
| `pulsating_bass_textures` | 10 | `sub_bass_fm_throbbing_saturator_v1` | 187.2 Hz | 0.6420 |
| `unstable_pitch_modulated_tones` | 10 | `chaotic_glissando_pitch_modulator_v1` | 395.6 Hz | 0.0577 |
| `irregular_rhythmic_transients` | 10 | `stochastic_percussive_transient_generator_v1` | 487.9 Hz | 0.2934 |
| `atmospheric_tension_beds` | 10 | `spectral_wind_tension_bed_v1` | 563.0 Hz | 0.5581 |
| `metallic_resonance` | 10 | `inharmonic_modal_plate_resonator_v1` | 674.6 Hz | 0.0191 |
| `synthetic_creature_sounds` | 10 | `formant_vocal_tract_creature_synth_v1` | 701.3 Hz | 0.0079 |
| `spectral_textures` | 10 | `additive_microtonal_spectral_bank_v1` | 1,090.8 Hz | 0.0082 |
| `glitching_machinery` | 10 | `stepped_buffer_stutter_machinery_v1` | 2,217.8 Hz | 0.0001 |
| `digital_interference` | 10 | `aliased_packet_corrupt_interference_v1` | 3,487.4 Hz | 0.0571 |

---

## Why this exists

General-purpose audio corpora (AudioSet, ESC-50, FSD50K) target everyday environmental sounds and have sparse, coarse coverage of cinematic horror sound design, non-linear synthesis textures, and tension/affective semantics. They also carry upstream copyright liability — scraped YouTube links, mixed CC-NC grants, untraceable provenance.

ZMH Horror Sound Semantics is the opposite bet on a small, defensible footprint:

- **Clean chain of custody.** Every waveform is synthesized mathematically with `numpy` / `scipy.signal`. No field recordings, no sample packs, no neural audio generators — so no third-party training-data laundering risk.
- **Exact ground truth.** Each clip ships its full DSP recipe (engine, seed, base frequency, modulation, filter, drive, reverb decay, envelopes), enabling parameter-conditioned and interpretable audio model research rather than black-label guessing.
- **A real emotion/tension ontology.** 8 affective labels, 5-point `intensity` and `tension` scales with paired `0.0–1.0` scores, 10 `cinematic_usage` deployment tags, spatial and tonal descriptors.
- **Truth in labeling.** All 100 v0.1 records are algorithmically proposed and explicitly marked `unreviewed_provisional`. No fabricated human judgments.
- **Human-in-the-loop, already wired.** A validated Label Studio interface plus pre-loaded predictions and a review-ingestion path that upgrades provenance when a human actually listens.

Primary target uses: horror/ambient sound classification, CLAP-style audio-text retrieval and contrastive pretraining, cinematic tension and affect modeling, sound-event temporal segmentation, sound-design recommendation, and game audio directors that map a live `0.0–1.0` tension score to a cue.

---

## Repository layout

```text
label-studio/                                # this repo = upstream platform + ZMH extension
├── README.md                                # this file
├── LICENSE / NOTICE                         # UPSTREAM Label Studio, Apache-2.0 (Heartex, Inc. / HumanSignal)
├── label_studio/ · web/ · deploy/ · docs/   # UPSTREAM platform sources (Apache-2.0, untouched)
├── pyproject.toml · Makefile · docker-compose.yml
└── zmh-datasets/
    └── horror-sound-semantics/              # ← THE PROJECT (Zazie-owned)
        ├── README.md                        # dataset-specific deep documentation
        ├── src/                             # generate → analyze → label → validate → export
        ├── configs/
        │   ├── annotation_config.xml        # waveform + semantic labeling interface (17 controls)
        │   └── dataset_schema.json          # JSON Schema Draft 2020-12, 38 properties
        ├── data/
        │   ├── demo_audio/                  # 100 × 16 kHz mono WAV, 8.4 MB
        │   ├── metadata.csv                 # 100 rows × 31 columns
        │   ├── training_data.jsonl          # 100 full structured ML records
        │   ├── provenance_manifest.csv      # per-sample IP & provenance tiers
        │   └── label_studio_tasks.json      # 100 importable tasks + predictions
        ├── docs/
        │   ├── DATASET_CARD.md              # datasheet, distributions, splits, acoustic profiles
        │   ├── ANNOTATION_GUIDELINES.md      # annotator handbook + vocabulary definitions
        │   ├── IP_OWNERSHIP.md              # 4-tier rights separation framework
        │   ├── QUALITY_REPORT.md            # validation results, test audit, limitations
        │   └── COMMERCIAL_LICENSE_DRAFT.md  # draft commercial license (not legal advice)
        └── tests/                           # 15 pytest tests
```

---

## Quickstart — dataset pipeline

CPU-only, `$0` budget, no GPU / Docker / model weights required.

```bash
# 1. Install the lightweight deps (numpy, scipy, jsonschema, requests, pytest)
cd zmh-datasets/horror-sound-semantics
pip install -r requirements.txt

# 2. Run the full pipeline: Generate → Analyze → Propose Labels → Validate → Export
python3 -m src.pipeline --count 100 --sample-rate 16000 --base-seed 202610

# 3. Verify the committed dataset independently
python3 -m src.validate_dataset

# 4. Run the test suite
python3 -m pytest -v
```

The pipeline is **idempotent**: it checks SHA-256 digests, never duplicates records, and preserves any `human_reviewed_verified` / `human_reviewed_corrected` annotations already ingested.

### Individual modules

```bash
# Synthesize procedural audio only (--sample-rate accepts 16000 | 22050 | 44100 | 48000)
python3 -m src.generate_audio --output-dir data/demo_audio --count 100 --sample-rate 16000

# Measure one file or a whole directory
python3 -m src.analyze_audio data/demo_audio/zmh_hss_v01_0001.wav

# Validate the Label Studio XML interface against the schema vocabularies
python3 -m src.label_studio_integration \
    --xml-config configs/annotation_config.xml \
    --schema configs/dataset_schema.json

# Full validation: schema, duplicates, WAV readability, SHA-256, split leakage
python3 -m src.validate_dataset

# Re-export JSONL, CSV, provenance manifest, Label Studio tasks
python3 -m src.export_dataset --input-jsonl data/training_data.jsonl --data-dir data
```

---

## Quickstart — annotation in Label Studio

You need a running Label Studio instance. This fork ships the full platform; pick the lightest option that works for you:

**Option A — pip (fastest for local review):**

```bash
# from the repository root
pip install .
label-studio start          # http://localhost:8080
```

**Option B — Docker (persistent volume for the DB and uploaded media):**

```bash
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data \
  heartexlabs/label-studio:latest
```

**Option C — this fork's own dev build** (when you need fork-local platform changes):

```bash
# from the repository root
pip install poetry && poetry install
make migrate-dev && make run-dev      # Django dev server on http://localhost:8080 (SQLite)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full upstream development workflow.

Then:

1. **Create the project** — paste `configs/annotation_config.xml` into **Settings → Labeling Interface → Code**, or call `LabelStudioClient.create_or_get_project()` in `src/label_studio_integration.py`.
2. **Import `data/label_studio_tasks.json`** — 100 tasks, each with `data` (metadata, synthesis parameters, measured features, audio path) and `predictions` (`model_version: "zmh-hss-provisional-v0.1"`) pre-populating all 17 controls. Serve `data/demo_audio/` over HTTP or sync it to local storage so `$audio_url` resolves.
3. **Review** — follow [docs/ANNOTATION_GUIDELINES.md](zmh-datasets/horror-sound-semantics/docs/ANNOTATION_GUIDELINES.md): listen in full, adjust waveform regions, correct semantic tags, rewrite the caption, drop the `[Provisional Algorithmic Caption]` prefix.
4. **Ingest reviews back into the dataset:**

   ```bash
   python3 -m src.pipeline --ingest-reviews path/to/label_studio_export.json
   ```

   `ingest_human_annotations_from_label_studio()` sets `human_review_status` to `human_reviewed_verified` or `human_reviewed_corrected` and promotes `annotation_provenance` to `human_expert_reviewed_v0.1` / `human_expert_corrected_v0.1`. Only after this step may any record be described as human-audited.

---

## Documentation

Start with the dataset's own [README](zmh-datasets/horror-sound-semantics/README.md), then:

| Document | Read it for |
|---|---|
| [DATASET_CARD.md](zmh-datasets/horror-sound-semantics/docs/DATASET_CARD.md) | Motivation, per-category distributions, splits, full schema and vocabulary reference |
| [ANNOTATION_GUIDELINES.md](zmh-datasets/horror-sound-semantics/docs/ANNOTATION_GUIDELINES.md) | Annotator handbook, controlled-vocabulary definitions, integrity rules |
| [QUALITY_REPORT.md](zmh-datasets/horror-sound-semantics/docs/QUALITY_REPORT.md) | Validation results, test audit, and documented technical limitations |
| [IP_OWNERSHIP.md](zmh-datasets/horror-sound-semantics/docs/IP_OWNERSHIP.md) | The 4-tier rights separation (upstream SW / ZMH software / compilations / noncopyrightable facts) |
| [COMMERCIAL_LICENSE_DRAFT.md](zmh-datasets/horror-sound-semantics/docs/COMMERCIAL_LICENSE_DRAFT.md) | Draft commercial licensing and positioning — **draft, not legal advice** |

Known v0.1 limitations are stated plainly in the quality report: zero human-reviewed annotations so far, procedural DSP rather than acoustic Foley (no bowed waterphone, no real room impulse responses), a 16 kHz mono baseline, and Label Studio integration validated via XML/schema and simulated export ingestion rather than a live multi-container deployment.

---

## License & intellectual property

This repository mixes rights regimes deliberately. See [IP_OWNERSHIP.md](zmh-datasets/horror-sound-semantics/docs/IP_OWNERSHIP.md) for the full framework.

- **Upstream Label Studio platform (Category A)** — all files outside `zmh-datasets/horror-sound-semantics/` remain **Apache License, Version 2.0**, © Heartex, Inc. / HumanSignal and contributors. Root [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) are preserved intact; `Label Studio` is a trademark of its owner. Forking conveys no ownership of upstream code, assets, or marks, and Zazie asserts none.
- **ZMH dataset tooling, schema, and configuration (Category B)** — `src/*.py`, `tests/*.py`, `configs/annotation_config.xml`, `configs/dataset_schema.json`: original works authored for Zazie Productions LLC.
- **ZMH creative content and compilations (Category C)** — documentation, taxonomy/ontology structure, curation and arrangement, and human-audited captions and annotations once reviewed.
- **Factual measurements and purely procedural waveforms (Category D)** — the 12 `acoustic_features` per clip are objective, noncopyrightable facts, and waveforms flagged `procedural_dsp_synthesis` do not automatically carry standalone sound-recording copyright under U.S. law. Commercialization is therefore structured around compilation/database rights, schema access, and data-supply services rather than overreaching copyright claims.

Third-party audio, sample packs, and pretrained generative model weights are **not** used anywhere in v0.1; every record's `provenance_manifest.csv` row carries the corresponding flags and SHA-256 checksum.

---

## About the upstream platform

<details>
<summary>What Label Studio is (upstream project this fork builds on)</summary>

<br>

Label Studio is an open source data labeling tool for audio, text, images, video, and time series, with a browser UI and export to common ML formats. It is used both to prepare raw data and to improve existing training data.

- Website: <https://labelstud.io/> · Docs: <https://labelstud.io/guide/> · Slack: <https://slack.labelstud.io/>
- Included labeling templates for many ML tasks, an ML-backend integration loop (pre-label → train → re-pre-label), and multi-user annotation with review.
- Deployment recipes that this fork inherits unchanged: **Docker Compose** (LS + Nginx + PostgreSQL, plus optional MinIO and MySQL overlays in `docker-compose.minio.yml` / `docker-compose.mysql.yml`), **poetry**, **conda**, and cloud templates (`azuredeploy.json`, `app.json`, `Dockerfile.heroku`, `Dockerfile.cloudrun`, `Dockerfile.hgface`).
- Upstream test suite: `make test` — runs `pytest label_studio -v -m "not integration_tests"` against SQLite via `uv`.
- Upstream contribution, security, and design-system guidance: [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [DESIGN.md](DESIGN.md).

Full upstream README history and marketing copy live in the [upstream repository](https://github.com/HumanSignal/label-studio) under Apache-2.0; this fork's README leads with the ZMH dataset because that is what this repository is for.

</details>

## Citation

If you build on this dataset, please cite both layers:

```bibtex
@misc{zmh_hss_2026,
  title  = {ZMH Horror Sound Semantics v0.1: A Provenance-Audited Cinematic Horror
            Audio Dataset with Audio-Language Semantics and a Label Studio Annotation Pipeline},
  author = {{Zazie Productions LLC — Zazie Media Holdings}},
  year   = {2026},
  note   = {100 procedurally synthesized clips, 10 categories, 38-field schema,
            zero third-party audio; all records unreviewed_provisional in v0.1}
}

@misc{label_studio,
  title  = {Label Studio},
  author = {HumanSignal},
  year   = {2019},
  url    = {https://github.com/HumanSignal/label-studio}
}
```
