# Intellectual Property, Provenance & Rights Separation Framework

**Organization:** Zazie Productions LLC — Zazie Media Holdings  
**Product:** ZMH Horror Sound Semantics (v0.1)  
**Manifest File:** `data/provenance_manifest.csv`

---

## 1. Executive Summary

To ensure clean commercialization, regulatory compliance, and respect for open-source licenses, **Zazie Media Holdings** enforces a strict four-tier separation of intellectual property across this repository.

**Critical Disclaimer:** Forking the `HumanSignal/label-studio` repository does **not** transfer any copyright, trademark, or patent rights in Label Studio to Zazie Productions LLC. All upstream Label Studio source code, assets, and documentation remain under the **Apache License, Version 2.0** and their original copyright holders (Heartex, Inc. / HumanSignal and contributors).

---

## 2. Four-Tier IP Classification Matrix

### Category A: Third-Party Open-Source Software (Upstream Label Studio)
- **Scope:** All files outside `zmh-datasets/horror-sound-semantics/`, including `label_studio/`, `web/`, `deploy/`, `docs/`, `LICENSE`, and `NOTICE`.
- **Owner:** Heartex, Inc. / HumanSignal and original open-source contributors.
- **Governing License:** Apache License, Version 2.0 (`LICENSE`).
- **ZMH Policy:**
  - All upstream copyright notices, `LICENSE`, and `NOTICE` files are preserved intact.
  - Zazie Productions LLC does **not** apply any proprietary or exclusive commercial license to Category A files.
  - Our dataset extension interacts with Label Studio solely through its documented XML configuration format (`configs/annotation_config.xml`), JSON task/prediction schemas (`data/label_studio_tasks.json`), and REST API endpoints (`src/label_studio_integration.py`).

### Category B: Zazie-Created Original Software & Schema Architecture
- **Scope:**
  - `zmh-datasets/horror-sound-semantics/src/*.py` (`generate_audio.py`, `analyze_audio.py`, `generate_metadata.py`, `label_studio_integration.py`, `validate_dataset.py`, `export_dataset.py`, `pipeline.py`)
  - `zmh-datasets/horror-sound-semantics/configs/annotation_config.xml`
  - `zmh-datasets/horror-sound-semantics/configs/dataset_schema.json`
  - `zmh-datasets/horror-sound-semantics/tests/*.py`
- **Owner:** Zazie Productions LLC — Zazie Media Holdings.
- **Rights Basis:** Original computer programs, taxonomy architecture, and schema definitions authored for Zazie Media Holdings.

### Category C: Zazie-Owned or Zazie-Controlled Creative Content & Compilations
- **Scope:**
  - Original human-authored documentation (`docs/*.md`, `README.md`).
  - Creative selection, coordination, taxonomy structure, and arrangement of the dataset parameters, controlled vocabularies, and multi-dimensional ontology (protectable as an original compilation/database structure to the extent permitted by applicable copyright law).
  - Human-authored and human-revised natural-language captions and verified perceptual annotations once audited via the Label Studio workflow (`human_review_status` in `["human_reviewed_verified", "human_reviewed_corrected"]`).
  - Future original human-performed instrument, vocal, or acoustic Foley recordings produced by Zazie Productions LLC (`generation_method = "human_performed_or_field_recording"`).
- **Owner:** Zazie Productions LLC — Zazie Media Holdings.

### Category D: Noncopyrightable or Uncertain Material
- **Scope:**
  1. **Factual Acoustic Measurements (`acoustic_features`):** Objective mathematical signal properties (`rms_energy`, `peak_amplitude`, `crest_factor_db`, `zero_crossing_rate`, `spectral_centroid_hz`, `spectral_bandwidth_hz`, `spectral_rolloff_85_hz`, `spectral_flatness`, `low_frequency_energy_ratio`, `high_frequency_energy_ratio`, `onset_strength_std`, `temporal_centroid_norm`). Under U.S. copyright law (*Feist Publications, Inc. v. Rural Telephone Service Co.*, 499 U.S. 340 (1991)), raw facts and objective measurements are not subject to standalone copyright protection.
  2. **Purely Procedurally Generated Waveforms (`data/demo_audio/*.wav`) and Unreviewed Algorithmic Metadata:** Under U.S. Copyright Office policy and federal precedent regarding machine/algorithmic outputs, audio files synthesized purely via deterministic mathematical execution of seeded DSP code (`generation_method = "procedural_dsp_synthesis"`) and unreviewed algorithmic labels (`annotation_provenance = "algorithmic_dsp_and_acoustic_heuristic_v0.1"`) do not automatically qualify for standalone copyright protection as sound recordings or literary works absent sufficient human creative authorship in the specific output.
- **ZMH Policy:**
  - Every record in `data/provenance_manifest.csv` explicitly flags `waveform_ip_category = "Category_D_Procedural_Synthetic_Waveform_Uncertain_Standalone_Copyright"` and `acoustic_features_ip_category = "Category_D_Factual_Signal_Measurements_Noncopyrightable"`.
  - Commercial licensing of v0.1 is structured around **compilation/database licensing, proprietary schema access, contractual data-delivery terms, and custom synthesis/curation services** (see `docs/COMMERCIAL_LICENSE_DRAFT.md`), rather than asserting standalone sound-recording copyright over purely procedural waveforms.

---

## 3. Provenance & Anti-Contamination Guarantees

Every sample in `data/provenance_manifest.csv` is verified against the following rules by `src/validate_dataset.py`:
- `uses_pretrained_ai_model = False` (Zero neural audio codec or diffusion weights used; eliminates third-party training-data laundering risks).
- `uses_third_party_samples = False` (Zero external sample libraries, Foley packs, or field recordings mixed into the synthesis chain).
- `is_authentic_human_or_field_recording = False` (Strict truth-in-labeling for procedural audio).
- SHA-256 cryptographic checksums (`sha256_checksum`) recorded for every `.wav` file to guarantee chain-of-custody verification.
