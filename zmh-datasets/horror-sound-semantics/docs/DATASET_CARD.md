# Dataset Card: ZMH Horror Sound Semantics (v0.1)

- **Dataset Name:** ZMH Horror Sound Semantics
- **Version:** 0.1
- **Publisher / Owner:** Zazie Productions LLC — Zazie Media Holdings
- **Domain:** Audio Machine Learning, Cinematic Sound Classification, Audio-Language Retrieval, Affective Sound Computing
- **Total Validated Samples:** 100 (`zmh_hss_v01_0001` – `zmh_hss_v01_0100`)
- **Audio Format:** 16,000 Hz, 16-bit PCM Mono WAV (`data/demo_audio/*.wav`)
- **Total Audio Duration:** 261.5 seconds (individual clips range from 2.00s to 3.25s)
- **Committed Audio Payload:** 8,372,400 bytes across 100 WAV files (≈ 8.4 MB / 7.98 MiB)
- **Structured Fields per Record:** 38 top-level schema properties (37 required), including 12 measured acoustic features, 14 DSP synthesis parameters, and 199 total `temporal_events` regions
- **Human-Reviewed Annotations in v0.1 MVP:** 0 (100% of v0.1 MVP records are explicitly marked `human_review_status: "unreviewed_provisional"` and `annotation_provenance: "algorithmic_dsp_and_acoustic_heuristic_v0.1"`)

---

## 1. Motivation & Target Applications

General-purpose audio datasets (such as AudioSet, ESC-50, and FSD50K) focus primarily on everyday environmental sound events (e.g., dog barking, sirens, footsteps) and have sparse, coarse coverage of cinematic horror sound design, non-linear synthesis textures, and tension/affective semantics.

**ZMH Horror Sound Semantics (v0.1)** establishes a domain-specific schema and reproducible baseline for:
1. **Horror Sound Classification:** Fine-grained 10-class recognition across tonal, percussive, glitch, creature, and atmospheric horror archetypes.
2. **Audio-to-Text Retrieval & Contrastive Audio-Language Pretraining (CLAP-style):** Pairing audio clips with synthesis- and feature-grounded natural-language captions (`natural_language_caption`).
3. **Cinematic Emotion & Tension Modeling:** Multi-dimensional affective tagging (`primary_emotion`, `secondary_emotions`, `intensity`, `intensity_score`, `tension`, `tension_score`).
4. **Sound-Event Identification & Temporal Segmentation:** Time-stamped region boundaries (`temporal_events`) compatible with Label Studio's `<Labels>` waveform interface.
5. **Experimental Music & Sound-Design Categorization:** Controlled descriptors for `acoustic_texture`, `rhythmic_characteristics`, `tonality`, and `perceived_spatial_properties`.
6. **Sound-Design Recommendation Systems:** Indexing sound cues by `cinematic_usage` (e.g., `jump_scare_stinger`, `slow_burn_dread_underscore`, `industrial_sci_fi_corridor`).

---

## 2. Dataset Composition & Sound-Event Categories

The v0.1 dataset contains **100 procedurally synthesized audio clips** evenly balanced across **10 sound-event categories** (10 samples per category: 5 synthesis recipes × 2 parameter/seed variants per recipe):

| Sound-Event Category | Count | DSP Synthesis Architecture (`engine_name`) | Mean Spectral Centroid (Hz) | Mean Spectral Flatness | Mean Low-Freq Ratio (<250 Hz) | Mean Crest Factor (dB) | Mean Onset Std |
|---|---:|---|---:|---:|---:|---:|---:|
| `dissonant_drones` | 10 | `detuned_tritone_cluster_drone_v1` | 143.0 | 0.0003 | 0.9108 | 10.01 | 0.0873 |
| `metallic_resonance` | 10 | `inharmonic_modal_plate_resonator_v1` | 674.6 | 0.0021 | 0.0191 | 12.85 | 0.1161 |
| `glitching_machinery` | 10 | `stepped_buffer_stutter_machinery_v1` | 2,217.8 | 0.0106 | 0.0001 | 14.97 | 0.1903 |
| `pulsating_bass_textures` | 10 | `sub_bass_fm_throbbing_saturator_v1` | 187.2 | 0.0000 | 0.6420 | 6.92 | 0.1294 |
| `digital_interference` | 10 | `aliased_packet_corrupt_interference_v1` | 3,487.4 | 0.7111 | 0.0571 | 3.79 | 0.0961 |
| `atmospheric_tension_beds` | 10 | `spectral_wind_tension_bed_v1` | 563.0 | 0.0009 | 0.5581 | 12.93 | 0.0225 |
| `irregular_rhythmic_transients` | 10 | `stochastic_percussive_transient_generator_v1` | 487.9 | 0.0105 | 0.2934 | 13.57 | 0.4785 |
| `synthetic_creature_sounds` | 10 | `formant_vocal_tract_creature_synth_v1` | 701.3 | 0.0011 | 0.0079 | 12.80 | 0.0645 |
| `spectral_textures` | 10 | `additive_microtonal_spectral_bank_v1` | 1,090.8 | 0.0076 | 0.0082 | 10.04 | 0.0374 |
| `unstable_pitch_modulated_tones` | 10 | `chaotic_glissando_pitch_modulator_v1` | 395.6 | 0.0023 | 0.0577 | 7.32 | 0.0622 |

---

## 3. Stratified Splits & Zero-Leakage Guarantee

Each category defines 5 distinct synthesis recipes (`{category}_r01` through `{category}_r05`), with 2 variants (`variant_index` 1 and 2) per recipe. Splits are assigned strictly at the **`recipe_id` group level**:
- **`train` (60 samples / 30 recipes):** Recipes `r01`, `r02`, `r03` across all 10 categories (6 samples per category).
- **`validation` (20 samples / 10 recipes):** Recipe `r04` across all 10 categories (2 samples per category).
- **`test` (20 samples / 10 recipes):** Recipe `r05` across all 10 categories (2 samples per category).

**Leakage Prevention:** Because both variants of any `recipe_id` are assigned to the same split, **0 recipes are shared across `train`, `validation`, and `test`** (`zero_recipe_leakage = True`, verified automatically in `src/validate_dataset.py`).

---

## 4. Schema & Controlled Vocabularies

Every record in `data/training_data.jsonl` is validated against `configs/dataset_schema.json` and includes:

- **Identification & Signal Specs:** `sample_id`, `audio_filename`, `duration`, `sample_rate`, `sha256_checksum`
- **Affective / Emotional Semantics:**
  - `primary_emotion` & `secondary_emotions`: `["dread", "anxiety", "uncertainty", "suspense", "shock", "unease", "anticipation", "disorientation"]`
  - `psychological_label_disclaimer`: Explicit notice distinguishing algorithmic provisional mappings from verified human listening judgments.
- **Acoustic & Musical Descriptors:**
  - `acoustic_texture`: `["dense_sub_harmonic", "inharmonic_metallic", "granular_stutter", "throbbing_low_end", "bitcrushed_aliased", "filtered_noise_wash", "percussive_erratic", "formant_guttural", "crystalline_spectral", "warped_microtonal"]`
  - `intensity`: `["subtle", "low", "moderate", "high", "extreme"]` + `intensity_score` (`0.0–1.0`)
  - `tension`: `["latent", "building", "sustained", "escalating", "climactic"]` + `tension_score` (`0.0–1.0`)
  - `rhythmic_characteristics`: `["arrhythmic_sustained", "slow_periodic_pulse", "fast_mechanical_stutter", "syncopated_heartbeat", "stochastic_intermittent", "accelerating_tremolo", "chaotic_polyrhythmic"]`
  - `tonality`: `["atonal_noise", "microtonal_cluster", "dissonant_bichord", "inharmonic_partial_series", "dark_pedal_drone", "unstable_glissando", "ambiguous_spectral"]`
  - `perceived_spatial_properties`: `["claustrophobic_intimate", "cavernous_reverberant", "distant_muffled", "wide_diffused_field", "focused_foreground_transient", "subterranean_rumble", "disembodied_floating"]`
  - `cinematic_usage`: Multi-label array from `["jump_scare_stinger", "slow_burn_dread_underscore", "psychological_breakdown", "industrial_sci_fi_corridor", "supernatural_manifestation", "monster_vocal_presence", "found_footage_signal_corrupt", "ritual_occult_ambience", "chase_sequence_pulse", "liminal_space_drone"]`
- **Provenance & Rights Status:**
  - `annotation_confidence`: Float (`0.0–1.0`)
  - `annotation_provenance`: `"algorithmic_dsp_and_acoustic_heuristic_v0.1"` (upgraded to `"human_expert_reviewed_v0.1"` or `"human_expert_corrected_v0.1"` only after Label Studio human review ingestion)
  - `human_review_status`: `"unreviewed_provisional"` (or `"human_reviewed_verified"` / `"human_reviewed_corrected"` / `"rejected"`)
  - `copyright_and_licensing_status`: `"procedural_audio_unverified_copyright_human_schema_zazie_owned"`
- **Nested Technical Objects:**
  - `synthesis_parameters`: Full DSP recipe parameters and random seed for 100% deterministic waveform regeneration.
  - `acoustic_features`: 12 measured spectral and temporal metrics (`rms_energy`, `peak_amplitude`, `crest_factor_db`, `zero_crossing_rate`, `spectral_centroid_hz`, `spectral_bandwidth_hz`, `spectral_rolloff_85_hz`, `spectral_flatness`, `low_frequency_energy_ratio`, `high_frequency_energy_ratio`, `onset_strength_std`, `temporal_centroid_norm`).
  - `temporal_events`: Time-stamped event segments (`start_sec`, `end_sec`, `event_label`).

---

## 5. Provenance, Ethics & Limitations

- **Zero Third-Party Audio or Pretrained Model Weights:** All 100 audio files in v0.1 are generated mathematically via `numpy` and `scipy.signal`. No copyrighted field recordings, sample packs, or neural audio generators were used.
- **No Fabricated Human Judgments:** All 100 records in v0.1 carry `human_review_status = "unreviewed_provisional"`. Emotional and psychological descriptors in v0.1 are provisional heuristic proposals awaiting human perceptual audit in Label Studio.
- **Procedural Audio Copyright Notice:** Purely procedural waveforms do not automatically qualify for standalone copyright protection under U.S. law; see `docs/IP_OWNERSHIP.md` for how Zazie Media Holdings separates compilation/schema rights from raw synthetic waveforms.
