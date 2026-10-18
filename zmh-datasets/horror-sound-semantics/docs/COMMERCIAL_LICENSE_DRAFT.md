# Commercial Data License Draft & Commercialization Strategy

> **LEGAL NOTICE — DRAFT ONLY (NOT FINALIZED):**  
> This document contains a preliminary draft commercial dataset license agreement and market positioning strategy prepared for internal business and legal review by **Zazie Productions LLC — Zazie Media Holdings**. It does **not** constitute formal legal advice or an executed contract and must be reviewed and finalized by qualified intellectual property counsel prior to commercial execution.

---

## Part I: Draft Commercial Dataset & Compilation License Agreement

### ZAZIE MEDIA HOLDINGS — PROPRIETARY DATASET & COMPILATION LICENSE AGREEMENT (DRAFT v0.1)

**Licensor:** Zazie Productions LLC, acting through its intellectual property division, Zazie Media Holdings ("Licensor" or "ZMH").  
**Product:** ZMH Horror Sound Semantics Dataset, Schema, and Curated Metadata Compilation (Version 0.1 and subsequent licensed releases, collectively the "Licensed Dataset Product").

#### 1. Explicit Scope of Rights & IP Carve-Outs
1.1 **Included Zazie-Owned / Zazie-Controlled Material (Categories B & C):** Subject to the terms of this Agreement and timely payment of applicable License Fees, Licensor grants Licensee a non-exclusive, non-transferable, non-sublicensable license to use:
- (a) The original **ZMH Horror Sound Semantics Taxonomy, JSON Schema (`dataset_schema.json`), and Controlled Vocabularies**;
- (b) The **selection, coordination, arrangement, and compilation** of dataset records, synthesis recipes, and parameter pairings (`training_data.jsonl`, `metadata.csv`, `provenance_manifest.csv`);
- (c) All **human-authored natural-language captions, expert perceptual annotations (`human_reviewed_verified` / `human_reviewed_corrected`), and documentation** owned by Licensor; and
- (d) Any **verified original human-performed instrument, vocal, or acoustic Foley recordings** explicitly designated in `provenance_manifest.csv` as `zazie_verified_original_recording_commercial_licensable`.

1.2 **Explicit Exclusion of Upstream Third-Party Open-Source Software (Category A):**  
This Agreement does **not** govern, re-license, or assert any proprietary ownership over **Label Studio** (`HumanSignal/label-studio`) or any third-party open-source software dependencies. Label Studio remains governed exclusively by the **Apache License, Version 2.0** (Copyright Heartex, Inc. / HumanSignal and contributors).

1.3 **Transparent Disclosure of Factual Measurements & Procedural Outputs (Category D):**  
Licensor expressly discloses that:
- (a) Objective acoustic signal measurements (`rms_energy`, `spectral_centroid_hz`, `zero_crossing_rate`, etc.) constitute unprotectable mathematical facts; and
- (b) Audio waveforms flagged in `provenance_manifest.csv` with `generation_method = "procedural_dsp_synthesis"` and `copyright_and_licensing_status = "procedural_audio_unverified_copyright_human_schema_zazie_owned"` are generated via deterministic mathematical signal processing and do not automatically carry standalone sound-recording copyright under U.S. law. Access to and commercial delivery of such procedural files under this Agreement are provided as part of Licensor's curated database compilation and contractual data-supply service.

#### 2. Permitted Commercial & AI-Training Uses
Depending on the License Tier purchased on the Order Form, Licensee may use the Licensed Dataset Product internally to:
- Train, fine-tune, evaluate, and benchmark machine-learning, deep-learning, and audio-language models (including audio classifiers, text-to-audio generators, video-to-audio synchronization models, and sound-design recommendation systems);
- Deploy trained model weights in commercial software products or production pipelines, provided that the trained model does not memorize or regurgitate substantial portions of Licensor's proprietary human-authored captions or compilation structure as a competing dataset product.

#### 3. Restrictions
Except as expressly permitted in writing by Licensor, Licensee shall not:
- Resell, redistribute, publish, or syndicate the raw Licensed Dataset Product (or any subset thereof) on public dataset hubs (including Hugging Face, Kaggle, GitHub, or Zenodo) or commercial sample-pack marketplaces;
- Remove, alter, or obscure provenance fields (`annotation_provenance`, `human_review_status`, `copyright_and_licensing_status`, `psychological_label_disclaimer`, or `provenance_manifest.csv`);
- Misrepresent procedurally synthesized audio (`procedural_dsp_synthesis`) as authentic human musical performance or field recordings, or misrepresent provisional algorithmic labels (`unreviewed_provisional`) as verified human psychological judgments.

#### 4. Warranties, Provenance Representation & Disclaimer
4.1 **Provenance Representation:** Licensor represents that the procedural audio samples in v0.1 (`generation_method = "procedural_dsp_synthesis"`) were synthesized without incorporating unlicensed third-party audio samples, copyrighted music recordings, or outputs of third-party generative AI models.  
4.2 **As-Is Warranty Disclaimer:** EXCEPT AS EXPRESSLY STATED IN SECTION 4.1, THE LICENSED DATASET PRODUCT IS PROVIDED "AS IS" WITHOUT WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR SPECIFIC MODEL ACCURACY OUTCOMES.

---

## Part II: Market Research & Commercialization Strategy

### 1. Comparable Datasets & Competitive Landscape

| Dataset | Publisher / Source | License | Size / Format | Primary Limitation for Commercial Horror / Cinematic AI |
|---|---|---|---|---|
| **AudioSet** | Google | CC BY 4.0 (YouTube URLs / embeddings only) | ~2M 10s clips (weak labels) | Distributes YouTube links/features rather than cleared WAV files; copyright-encumbered source videos; coarse ontology lacking cinematic tension/emotion/DSP parameters. |
| **ESC-50** | Karol Piczak | CC BY-NC 3.0 (Non-Commercial) | 2,000 5s clips (50 classes) | Non-commercial restriction on primary subset; environmental everyday sounds rather than cinematic horror textures. |
| **FSD50K** | UPF / Freesound | Mixed CC0, CC BY, CC BY-NC | 51,197 clips (200 AudioSet classes) | Heterogeneous licensing across contributors; noisy crowd-sourced tags; lacks structured cinematic emotion/tension ratings and synthesis provenance. |
| **AudioCaps / WavCaps** | Academic / Multi-source | Academic / Research-only or Mixed CC | 46k–400k captioned clips | Captions are general event descriptions (often LLM-rewritten from Freesound/AudioSet tags); legally risky for commercial foundation model training due to YouTube/NC upstream sources. |
| **Clotho** | Tampere Univ. | Academic / Freesound CC licenses | 4,981 clips (5 captions each) | Focused on ambient environmental scenes; lacks horror-specific taxonomy, synthesis ground truth, and commercial indemnification clarity. |

### 2. Commercial Differentiation of ZMH Horror Sound Semantics
1. **100% Clean Chain-of-Custody (Zero Scraping / Zero Third-Party Copyright Contamination):** Every sample in v0.1 is generated from first-principles mathematical DSP (with planned v0.2+ expansion into Zazie-owned original acoustic Foley recordings), eliminating the YouTube/CC-BY-NC copyright liability that plagues public audio-language datasets.
2. **Multi-Dimensional Cinematic & Affective Ontology:** Combines 10 horror sound-event classes, 10 acoustic textures, 8 affective emotion tags, continuous and categorical intensity/tension scores, rhythmic/tonal/spatial descriptors, cinematic usage tags, time-aligned region segments, and natural-language captions.
3. **Exact Physical & Synthesis Ground Truth:** Unlike scraped datasets where the underlying signal generation process is unknown, every procedural clip includes exact DSP synthesis parameters (`base_frequency_hz`, `modulation_rate_hz`, `filter_cutoff_hz`, `reverb_decay_sec`, `random_seed`) alongside 12 measured spectral/temporal features—ideal for training **controllable/parameter-conditioned neural audio synthesizers** and **interpretable audio encoders**.
4. **Strict Provenance & Review Transparency:** Explicitly separates algorithmic provisional labels (`unreviewed_provisional`) from human-audited labels (`human_reviewed_verified` / `human_reviewed_corrected`).

### 3. Target Buyer Personas & ML Use Cases
1. **Generative Audio & Video-to-Audio AI Developers:** Companies building text-to-SFX, video-to-foley, and controllable neural synthesizer models needing copyright-clean audio paired with rich timbral, affective, and parameter metadata.
2. **Game Studios & Interactive Audio Middleware Vendors:** Teams developing adaptive horror/sci-fi game audio engines (Wwise/FMOD plugins, procedural threat-level audio directors) that map real-time gameplay tension/intensity scores (`0.0–1.0`) to synthesis parameters and sound cues.
3. **Film/Trailer Post-Production & Stock Audio Catalog Platforms:** Companies building semantic audio search, auto-tagging, and mood-based recommendation systems for editors scoring horror, thriller, and sci-fi content.
4. **Academic & Industrial Affective Computing Labs:** Researchers studying acoustic correlates of perceived tension, dread, startle (`crest_factor_db`, `onset_strength_std`), and psychoacoustic dissonance.

### 4. Recommended Data Packaging & Distribution Channels
- **Data Formats:**
  - `training_data.jsonl` (nested JSON Lines for PyTorch / Hugging Face Datasets / WebDataset pipelines)
  - `metadata.csv` + `provenance_manifest.csv` (tabular analytics, Pandas / DuckDB / Parquet conversion)
  - `label_studio_tasks.json` + `annotation_config.xml` (turnkey human-in-the-loop annotation package)
  - Audio delivered as 16 kHz mono WAV (ML baseline tier) and 48 kHz / 24-bit WAV or lossless FLAC (Pro Studio / Generative Audio tier).
- **Distribution Marketplaces (Subject to Executive Approval Before Publication):**
  - Direct B2B enterprise licensing via Zazie Media Holdings
  - AWS Data Exchange / Snowflake Marketplace / Databricks Marketplace (for enterprise AI teams)
  - Dewey Data / Defined.ai /isser-style curated AI training data marketplaces
  - Public preview subset (e.g., 10–20 watermarked or low-rate preview records) for technical evaluation only.

---

## Part III: Draft Product Listing & Preliminary Pricing Strategy

### Draft Product Listing Copy (For Internal Review — Do Not Publish Without Approval)

> **Product Title:** ZMH Horror Sound Semantics (v0.1 MVP / Enterprise Custom Editions)  
> **Publisher:** Zazie Productions LLC — Zazie Media Holdings  
> **Summary:** A provenance-audited dataset and parameter-grounded synthesis corpus for cinematic horror audio classification, affective sound computing, and controllable text-to-SFX machine learning.  
> **Key Specifications (v0.1 MVP Baseline):**
> - **100 Validated Audio Clips** across 10 horror sound-event archetypes (`dissonant_drones`, `metallic_resonance`, `glitching_machinery`, `pulsating_bass_textures`, `digital_interference`, `atmospheric_tension_beds`, `irregular_rhythmic_transients`, `synthetic_creature_sounds`, `spectral_textures`, `unstable_pitch_modulated_tones`).
> - **Rich 38-Field Semantic Schema** including primary/secondary emotions (`dread`, `anxiety`, `uncertainty`, `suspense`, `shock`, `unease`, `anticipation`, `disorientation`), intensity & tension ratings, rhythmic/tonal/spatial descriptors, cinematic usage tags, time-aligned region boundaries, and natural-language captions.
> - **Complete DSP & Acoustic Ground Truth:** Includes exact synthesis parameters, random seeds, SHA-256 checksums, and 12 measured spectral/temporal signal metrics per clip.
> - **Clean Provenance & Zero Third-Party Sample Contamination:** 100% free of scraped YouTube audio, unlicensed sample packs, or third-party generative AI outputs. Includes full per-sample `provenance_manifest.csv`.

### Preliminary Pricing Strategy (Subject to Validation & Expansion)

*Note: Pricing below is a preliminary structural proposal for Zazie Media Holdings and makes no fabricated claims regarding current market demand or guaranteed sales volume.*

| Tier | Target Customer | Deliverable Scope | Preliminary Pricing Model |
|---|---|---|---|
| **Tier 1: Evaluation / MVP Developer Preview (v0.1)** | ML engineers & R&D teams evaluating schema compatibility | 100 procedural samples (16 kHz WAV), full schema, JSONL/CSV exports, provisional algorithmic annotations (`unreviewed_provisional`) | **$0 – $250** (Evaluation license / B2B lead magnet for custom orders) |
| **Tier 2: Pro Studio & Human-Verified Edition (Planned v1.0)** | Indie/Mid-market AI audio startups & game audio R&D teams | 1,000–2,500 samples (48 kHz WAV) combining procedural DSP + original Zazie-recorded acoustic Foley, **100% human-reviewed annotations & captions** (`human_reviewed_verified`) | **$2,500 – $7,500** per annual commercial R&D / model-training seat |
| **Tier 3: Enterprise Foundation Model & Custom Synthesis License** | Commercial generative AI labs, middleware vendors & major studios | 10,000+ custom-parameterized clips + on-demand deterministic synthesis engine access + custom taxonomy extensions + full legal provenance audit package | **$15,000 – $50,000+** (Custom Enterprise Data & Services Agreement) |
