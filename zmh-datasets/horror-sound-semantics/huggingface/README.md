---
pretty_name: ZMH Horror Sound Semantics (v0.1)
license: other
license_name: zmh-private-evaluation-no-license-selected
license_link: LICENSE.md
language:
- en
language_creators:
- machine-generated
annotations_creators:
- machine-generated
multilinguality:
- monolingual
source_datasets:
- original
size_categories:
- n<1K
task_categories:
- audio-classification
task_ids:
- audio-emotion-recognition
tags:
- horror
- sound-design
- cinematic-audio
- sound-effects
- synthetic
- procedural-audio
- audio-captioning
- sound-event-detection
- affective-computing
- provenance
- unreviewed-annotations
- label-studio
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*
  - split: validation
    path: data/validation/*
  - split: test
    path: data/test/*
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: sample_id
    dtype: string
  - name: audio_filename
    dtype: string
  - name: dataset_name
    dtype: string
  - name: dataset_version
    dtype: string
  - name: publisher
    dtype: string
  - name: split
    dtype: string
  - name: sound_event_category
    dtype: string
  - name: natural_language_caption
    dtype: string
  - name: primary_emotion
    dtype: string
  - name: secondary_emotions
    list: string
  - name: acoustic_texture
    dtype: string
  - name: intensity
    dtype: string
  - name: intensity_score
    dtype: float64
  - name: tension
    dtype: string
  - name: tension_score
    dtype: float64
  - name: rhythmic_characteristics
    dtype: string
  - name: tonality
    dtype: string
  - name: perceived_spatial_properties
    dtype: string
  - name: cinematic_usage
    list: string
  - name: duration
    dtype: float64
  - name: sample_rate
    dtype: int64
  - name: sha256_checksum
    dtype: string
  - name: recipe_id
    dtype: string
  - name: variant_index
    dtype: int64
  - name: random_seed
    dtype: int64
  - name: generation_method
    dtype: string
  - name: uses_pretrained_ai_model
    dtype: bool
  - name: uses_third_party_samples
    dtype: bool
  - name: is_authentic_human_or_field_recording
    dtype: bool
  - name: annotation_confidence
    dtype: float64
  - name: annotation_provenance
    dtype: string
  - name: human_review_status
    dtype: string
  - name: copyright_and_licensing_status
    dtype: string
  - name: psychological_label_disclaimer
    dtype: string
  - name: content_origin
    dtype: string
  - name: dsp_engine_name
    dtype: string
  - name: waveform_ip_category
    dtype: string
  - name: acoustic_features_ip_category
    dtype: string
  - name: schema_and_code_ip_category
    dtype: string
  - name: curation_and_selection_ip_category
    dtype: string
  - name: upstream_platform_ip_category
    dtype: string
  - name: commercial_licensing_eligibility
    dtype: string
  - name: synthesis_parameters
    struct:
    - name: engine_name
      dtype: string
    - name: recipe_id
      dtype: string
    - name: random_seed
      dtype: int64
    - name: base_frequency_hz
      dtype: float64
    - name: modulation_rate_hz
      dtype: float64
    - name: modulation_depth
      dtype: float64
    - name: filter_type
      dtype: string
    - name: filter_cutoff_hz
      dtype: float64
    - name: noise_mix_ratio
      dtype: float64
    - name: nonlinear_drive
      dtype: float64
    - name: reverb_decay_sec
      dtype: float64
    - name: envelope_attack_sec
      dtype: float64
    - name: envelope_release_sec
      dtype: float64
    - name: additional_params_json
      dtype: string
  - name: acoustic_features
    struct:
    - name: rms_energy
      dtype: float64
    - name: peak_amplitude
      dtype: float64
    - name: crest_factor_db
      dtype: float64
    - name: zero_crossing_rate
      dtype: float64
    - name: spectral_centroid_hz
      dtype: float64
    - name: spectral_bandwidth_hz
      dtype: float64
    - name: spectral_rolloff_85_hz
      dtype: float64
    - name: spectral_flatness
      dtype: float64
    - name: low_frequency_energy_ratio
      dtype: float64
    - name: high_frequency_energy_ratio
      dtype: float64
    - name: onset_strength_std
      dtype: float64
    - name: temporal_centroid_norm
      dtype: float64
  - name: temporal_events
    list:
    - name: start_sec
      dtype: float64
    - name: end_sec
      dtype: float64
    - name: event_label
      dtype: string
  splits:
  - name: train
    num_bytes: 10724290
    num_examples: 120
  - name: validation
    num_bytes: 3463742
    num_examples: 40
  - name: test
    num_bytes: 3447822
    num_examples: 40
  download_size: 17635854
  dataset_size: 17635854

---

# ZMH Horror Sound Semantics (v0.1)

**Publisher / rights holder:** Zazie Productions LLC — Zazie Media Holdings
**Dataset version:** 0.1 (MVP) · **200 clips** · **10 cinematic horror sound-event categories** · **0 human-reviewed records**
**Source repository:** `zazieproductions/ZMH-Horror-Sound-Semantics` (GitHub, private) — `zmh-datasets/horror-sound-semantics/`
**Repository visibility on the Hub:** private, evaluation only — see [Licensing](#licensing-and-redistribution-status)

> **Read this first — status of the annotations.**
> Every one of the 200 records carries `human_review_status = "unreviewed_provisional"` and
> `annotation_provenance = "algorithmic_dsp_and_acoustic_heuristic_v0.1"`. All semantic labels
> (emotion, tension, intensity, texture, cinematic usage), every natural-language caption and every
> temporal region were produced by deterministic heuristics from the synthesis parameters and measured
> acoustic features. They are **not expert-validated**, **not human-verified**, and must not be
> described or used as human listening judgments until they have been reviewed in Label Studio and the
> `human_review_status` field has been updated to `human_reviewed_verified` or `human_reviewed_corrected`.

## Dataset summary

ZMH Horror Sound Semantics is a small, fully reproducible audio-semantics dataset for cinematic horror sound design.
It pairs 200 original, procedurally synthesized horror sound clips with a domain-specific ontology
(10 sound-event categories, 8 affective tags, 10 acoustic textures, 10 cinematic-usage tags), 12 measured acoustic
features per clip, a provisional natural-language caption per clip, time-aligned event regions, full DSP synthesis
parameters for byte-exact regeneration, and a per-sample intellectual-property / provenance classification.

The dataset was built as an isolated extension inside Zazie Productions LLC's fork of
[Label Studio](https://github.com/HumanSignal/label-studio) (upstream platform: Apache-2.0) so that the provisional
labels can be audited by ear. The Label Studio labeling configuration used for that review is shipped under
`supplementary/`.

### Measured properties of this Hub repository

| Property | Verified value |
|---|---|
| Rows (`train` / `validation` / `test`) | 200 (120 / 40 / 40) |
| Audio files | 200 × 16,000 Hz, 16-bit PCM mono WAV (`data/<split>/zmh_hss_v01_NNNN.wav`) |
| Total / min / max clip duration | 525.0 s / 2.00 s / 3.25 s (319.5 s train · 103.0 s validation · 102.5 s test) |
| Audio payload | 16,808,800 bytes (≈ 16.0 MiB) |
| Sound-event categories | 10, exactly 20 clips each (12 train / 4 validation / 4 test per category) |
| Distinct synthesis recipes (`recipe_id`) | 100 (10 per category), 2 seed variants each; **0 recipes shared across splits** |
| Columns per row (after `datasets` loads the audio) | 46 (see [Data fields](#data-fields)) |
| Natural-language captions | 200, all distinct, all prefixed `[Provisional Algorithmic Caption]` |
| `temporal_events` regions | 450 |
| `human_review_status` | `unreviewed_provisional` for 200 / 200 |
| `annotation_provenance` | `algorithmic_dsp_and_acoustic_heuristic_v0.1` for 200 / 200 |
| `generation_method` | `procedural_dsp_synthesis` for 200 / 200 |
| `uses_pretrained_ai_model` / `uses_third_party_samples` / `is_authentic_human_or_field_recording` | `false` / `false` / `false` for 200 / 200 |
| Third-party / imported records | **0** (none approved; none included) |

## Content origin: what is original, what is algorithmic, what is third party

This repository deliberately keeps three kinds of material separate. Every row carries machine-readable fields
that encode the classification below; the full framework is in `supplementary/IP_OWNERSHIP.md` and the per-sample
manifest in `supplementary/provenance_manifest.csv`.

| Material | What it is | How it is marked in the rows | Origin / rights position |
|---|---|---|---|
| **Original ZMH audio** | 200 waveforms synthesized from seeded `numpy` / `scipy.signal` DSP code written for Zazie. No samples, field recordings, or pretrained generative weights were used. | `content_origin = "zmh_original_procedural_dsp_audio"`, `generation_method = "procedural_dsp_synthesis"`, the three provenance booleans all `false`, `sha256_checksum`, `synthesis_parameters` | Zazie-controlled. Note the deliberate caveat recorded in `waveform_ip_category`: purely procedural waveforms do not automatically qualify for standalone sound-recording copyright under U.S. law, so Zazie's claims are scoped to the compilation, schema, curation, and documentation (`curation_and_selection_ip_category`, `schema_and_code_ip_category`). |
| **Original ZMH schema, taxonomy, curation and documentation** | The controlled vocabularies, JSON Schema, Label Studio config, recipe design and selection, and this card. | `schema_and_code_ip_category = "Category_B_…"`, `curation_and_selection_ip_category = "Category_C_…"` | Zazie Productions LLC — Zazie Media Holdings. |
| **Algorithmically generated metadata (unreviewed)** | `primary_emotion`, `secondary_emotions`, `intensity*`, `tension*`, `acoustic_texture`, `rhythmic_characteristics`, `tonality`, `perceived_spatial_properties`, `cinematic_usage`, `natural_language_caption`, `temporal_events`, `annotation_confidence`. | `annotation_provenance = "algorithmic_dsp_and_acoustic_heuristic_v0.1"`, `human_review_status = "unreviewed_provisional"`, `psychological_label_disclaimer` | Heuristic proposals awaiting human perceptual audit. Not human judgments; not expert-validated. |
| **Factual acoustic measurements** | The 12 values in `acoustic_features`, computed by windowed FFT/STFT and envelope analysis. | `acoustic_features_ip_category = "Category_D_Factual_Signal_Measurements_Noncopyrightable"` | Objective measurements; treated as noncopyrightable facts. |
| **Third-party licensed content** | **None.** No external datasets, sample packs, recordings, or Hub datasets are included in this repository. | Would require `uses_third_party_samples = true` and a separate config with attribution columns — no such rows exist. | Any future third-party import requires separate, explicit approval by the rights holder and would be published as its own config with its own licence and attribution, never merged into this one. |
| **Upstream Label Studio platform** | Not included here. The annotation tooling lives in the source repository under its upstream Apache-2.0 licence. | `upstream_platform_ip_category = "Category_A_Label_Studio_Apache_2_0_Heartex_HumanSignal"` | © Heartex, Inc. / HumanSignal and contributors. Zazie asserts no ownership of upstream code, assets, or trademarks. |

## Supported tasks

The schema was designed for the following uses. All of them should be treated as **experiments on provisional
labels** until the human review pass is complete.

1. **Horror sound-event classification** — 10-class `sound_event_category` (the only label that is directly determined by the synthesis recipe rather than inferred by a heuristic).
2. **Audio–text retrieval / contrastive audio-language pretraining** — `audio` ↔ `natural_language_caption`.
3. **Cinematic affect and tension modeling** — `primary_emotion`, `secondary_emotions`, `intensity(_score)`, `tension(_score)`.
4. **Sound-event temporal segmentation** — `temporal_events` (`start_sec`, `end_sec`, `event_label`).
5. **Sound-design descriptor prediction** — `acoustic_texture`, `rhythmic_characteristics`, `tonality`, `perceived_spatial_properties`, `cinematic_usage`.
6. **Acoustic-feature regression / sanity baselines** — `acoustic_features`.

## Dataset structure

### Layout

```text
.
├── README.md                          # this card
├── LICENSE.md                         # licensing status notice (no public licence selected)
├── MANIFEST.sha256                    # SHA-256 of every file in this repository
├── data/
│   ├── train/       metadata.jsonl + 120 WAV
│   ├── validation/  metadata.jsonl +  40 WAV
│   └── test/        metadata.jsonl +  40 WAV
└── supplementary/
    ├── provenance_manifest.csv        # per-sample IP / provenance / review-status manifest
    ├── dataset_schema.json            # JSON Schema (Draft 2020-12) of the source records
    ├── label_studio_annotation_config.xml
    ├── IP_OWNERSHIP.md
    └── ANNOTATION_GUIDELINES.md
```

The repository follows the Hub **AudioFolder** convention: each `metadata.jsonl` row has a `file_name` that points
at the WAV in the same directory, and `datasets` exposes it as the `audio` column.

### Loading

```python
from datasets import load_dataset

ds = load_dataset("<namespace>/<repo>")            # private repo: pass token=... or run `huggingface-cli login`
row = ds["train"][0]
row["audio"]                                       # decoded waveform + sampling_rate
row["sound_event_category"], row["natural_language_caption"], row["human_review_status"]
```

### Data instance

```json
{
  "audio": {"path": "zmh_hss_v01_0001.wav", "sampling_rate": 16000, "array": "..."},
  "sample_id": "zmh_hss_v01_0001",
  "split": "train",
  "sound_event_category": "dissonant_drones",
  "natural_language_caption": "[Provisional Algorithmic Caption] A moderate-intensity, building-tension detuned tritone and minor-second oscillator cluster drone (2.50s at 16000 Hz) centered near 62.8 Hz ...",
  "primary_emotion": "suspense",
  "secondary_emotions": ["unease", "uncertainty"],
  "acoustic_texture": "dense_sub_harmonic",
  "intensity": "moderate", "intensity_score": 0.4661,
  "tension": "building", "tension_score": 0.2598,
  "cinematic_usage": ["liminal_space_drone", "ritual_occult_ambience"],
  "duration": 2.5, "sample_rate": 16000,
  "sha256_checksum": "087bea8a…",
  "recipe_id": "dissonant_drones_r01", "variant_index": 1, "random_seed": 202711,
  "generation_method": "procedural_dsp_synthesis",
  "uses_pretrained_ai_model": false, "uses_third_party_samples": false, "is_authentic_human_or_field_recording": false,
  "annotation_confidence": 0.74,
  "annotation_provenance": "algorithmic_dsp_and_acoustic_heuristic_v0.1",
  "human_review_status": "unreviewed_provisional",
  "copyright_and_licensing_status": "procedural_audio_unverified_copyright_human_schema_zazie_owned",
  "content_origin": "zmh_original_procedural_dsp_audio",
  "waveform_ip_category": "Category_D_Procedural_Synthetic_Waveform_Uncertain_Standalone_Copyright",
  "synthesis_parameters": {"engine_name": "detuned_tritone_cluster_drone_v1", "base_frequency_hz": 62.75, "...": "...", "additional_params_json": "{\"interval_ratios\": [0.5, 1.0, 1.0667, 1.4142, 1.498]}"},
  "acoustic_features": {"rms_energy": 0.287759, "spectral_centroid_hz": 92.6, "...": "..."},
  "temporal_events": [{"start_sec": 0.0, "end_sec": 2.5, "event_label": "dissonant_drones"}]
}
```

### Data fields

**Identification & signal**

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio` | 16 kHz, 16-bit PCM mono WAV, decoded on access. |
| `sample_id` / `audio_filename` | string | `zmh_hss_v01_0001` … `zmh_hss_v01_0200`; filename = `sample_id` + `.wav`. |
| `dataset_name`, `dataset_version`, `publisher` | string | Constant: `ZMH Horror Sound Semantics`, `0.1`, `Zazie Productions LLC — Zazie Media Holdings`. |
| `split` | string | `train` / `validation` / `test` (redundant with the Hub split, kept for integrity checks). |
| `duration`, `sample_rate` | float64, int64 | Seconds; always 16000 Hz. |
| `sha256_checksum` | string | SHA-256 of the WAV bytes; identical to the value in the source repository and `supplementary/provenance_manifest.csv`. |

**Semantic labels — all provisional / algorithmic in v0.1**

| Field | Type | Controlled vocabulary |
|---|---|---|
| `sound_event_category` | string | `dissonant_drones`, `metallic_resonance`, `glitching_machinery`, `pulsating_bass_textures`, `digital_interference`, `atmospheric_tension_beds`, `irregular_rhythmic_transients`, `synthetic_creature_sounds`, `spectral_textures`, `unstable_pitch_modulated_tones` |
| `primary_emotion`, `secondary_emotions` | string, list[string] | `dread`, `anxiety`, `uncertainty`, `suspense`, `shock`, `unease`, `anticipation`, `disorientation` |
| `acoustic_texture` | string | `dense_sub_harmonic`, `inharmonic_metallic`, `granular_stutter`, `throbbing_low_end`, `bitcrushed_aliased`, `filtered_noise_wash`, `percussive_erratic`, `formant_guttural`, `crystalline_spectral`, `warped_microtonal` |
| `intensity` + `intensity_score` | string + float64 | `subtle`, `low`, `moderate`, `high`, `extreme`; score in `[0, 1]` |
| `tension` + `tension_score` | string + float64 | `latent`, `building`, `sustained`, `escalating`, `climactic`; score in `[0, 1]` |
| `rhythmic_characteristics` | string | `arrhythmic_sustained`, `slow_periodic_pulse`, `fast_mechanical_stutter`, `syncopated_heartbeat`, `stochastic_intermittent`, `accelerating_tremolo`, `chaotic_polyrhythmic` |
| `tonality` | string | `atonal_noise`, `microtonal_cluster`, `dissonant_bichord`, `inharmonic_partial_series`, `dark_pedal_drone`, `unstable_glissando`, `ambiguous_spectral` |
| `perceived_spatial_properties` | string | `claustrophobic_intimate`, `cavernous_reverberant`, `distant_muffled`, `wide_diffused_field`, `focused_foreground_transient`, `subterranean_rumble`, `disembodied_floating` |
| `cinematic_usage` | list[string] | `jump_scare_stinger`, `slow_burn_dread_underscore`, `psychological_breakdown`, `industrial_sci_fi_corridor`, `supernatural_manifestation`, `monster_vocal_presence`, `found_footage_signal_corrupt`, `ritual_occult_ambience`, `chase_sequence_pulse`, `liminal_space_drone` |
| `natural_language_caption` | string | Template-generated from synthesis parameters and measured features; prefixed `[Provisional Algorithmic Caption]`. The prefix is removed only when a human reviewer rewrites the caption. |
| `temporal_events` | list[struct{`start_sec`, `end_sec`, `event_label`}] | Heuristic event regions (1–17 per clip, 450 total). |
| `annotation_confidence` | float64 | Heuristic self-estimate in `[0.74, 0.80]`; **not** inter-annotator agreement. |
| `psychological_label_disclaimer` | string | Per-row restatement that affective labels are algorithmic until reviewed. |

**Provenance, review status & rights**

| Field | Type | Values in v0.1 |
|---|---|---|
| `annotation_provenance` | string | `algorithmic_dsp_and_acoustic_heuristic_v0.1` (200/200). Becomes `human_expert_reviewed_v0.1` / `human_expert_corrected_v0.1` only after Label Studio review ingestion. |
| `human_review_status` | string | `unreviewed_provisional` (200/200). Other schema values: `human_reviewed_verified`, `human_reviewed_corrected`, `rejected`. |
| `copyright_and_licensing_status` | string | `procedural_audio_unverified_copyright_human_schema_zazie_owned` (200/200). |
| `content_origin` | string | `zmh_original_procedural_dsp_audio` (200/200). |
| `generation_method` | string | `procedural_dsp_synthesis` (200/200). |
| `uses_pretrained_ai_model`, `uses_third_party_samples`, `is_authentic_human_or_field_recording` | bool | `false` (200/200). |
| `recipe_id`, `variant_index`, `random_seed` | string, int64, int64 | Recipe `{category}_r01…r10`, variant 1–2, seed for byte-exact regeneration. |
| `dsp_engine_name` | string | Synthesis engine (one per category; see table below). |
| `waveform_ip_category` | string | `Category_D_Procedural_Synthetic_Waveform_Uncertain_Standalone_Copyright` |
| `acoustic_features_ip_category` | string | `Category_D_Factual_Signal_Measurements_Noncopyrightable` |
| `schema_and_code_ip_category` | string | `Category_B_Zazie_Original_Software_And_Taxonomy` |
| `curation_and_selection_ip_category` | string | `Category_C_Zazie_Controlled_Dataset_Compilation_And_Parameter_Curation` |
| `upstream_platform_ip_category` | string | `Category_A_Label_Studio_Apache_2_0_Heartex_HumanSignal` |
| `commercial_licensing_eligibility` | string | `Licensable_Compilation_And_Data_Delivery_With_Procedural_Audio_Disclosure` |

**Nested technical objects**

| Field | Type | Description |
|---|---|---|
| `synthesis_parameters` | struct | `engine_name`, `recipe_id`, `random_seed`, `base_frequency_hz`, `modulation_rate_hz`, `modulation_depth`, `filter_type`, `filter_cutoff_hz`, `noise_mix_ratio`, `nonlinear_drive`, `reverb_decay_sec`, `envelope_attack_sec`, `envelope_release_sec`, plus `additional_params_json` — the engine-specific extras (e.g. `interval_ratios`, `formant_f1_hz`, `quantization_levels`) serialised as a JSON string because their keys differ per engine. |
| `acoustic_features` | struct of 12 float64 | `rms_energy`, `peak_amplitude`, `crest_factor_db`, `zero_crossing_rate`, `spectral_centroid_hz`, `spectral_bandwidth_hz`, `spectral_rolloff_85_hz`, `spectral_flatness`, `low_frequency_energy_ratio`, `high_frequency_energy_ratio`, `onset_strength_std`, `temporal_centroid_norm`. |

### Splits and leakage guarantee

Each category has 10 synthesis recipes (`r01`–`r10`) with 2 seed variants each. Splits are assigned at the
**recipe level**, so both variants of a recipe always land in the same split:

| Split | Rows | Recipes per category | Recipe ids | Duration |
|---|---:|---:|---|---:|
| `train` | 120 | 6 | `r01`, `r02`, `r03`, `r06`, `r07`, `r08` | 319.5 s |
| `validation` | 40 | 2 | `r04`, `r09` | 103.0 s |
| `test` | 40 | 2 | `r05`, `r10` | 102.5 s |

0 recipes are shared across splits (`zero_recipe_leakage = true`, checked by the source repository's validator).

### Per-category acoustic profile (means over the 20 clips per category)

| Sound-event category | Clips | DSP engine (`dsp_engine_name`) | Spectral centroid (Hz) | Spectral flatness | Low-freq ratio (<250 Hz) | Crest factor (dB) | Onset std |
|---|---:|---|---:|---:|---:|---:|---:|
| `dissonant_drones` | 20 | `detuned_tritone_cluster_drone_v1` | 209.3 | 0.0014 | 0.8648 | 9.78 | 0.0771 |
| `metallic_resonance` | 20 | `inharmonic_modal_plate_resonator_v1` | 943.5 | 0.0176 | 0.0124 | 12.51 | 0.1080 |
| `glitching_machinery` | 20 | `stepped_buffer_stutter_machinery_v1` | 2,867.5 | 0.0118 | 0.0000 | 14.84 | 0.1741 |
| `pulsating_bass_textures` | 20 | `sub_bass_fm_throbbing_saturator_v1` | 236.7 | 0.0000 | 0.5849 | 6.56 | 0.1197 |
| `digital_interference` | 20 | `aliased_packet_corrupt_interference_v1` | 3,497.1 | 0.7057 | 0.0572 | 3.74 | 0.0824 |
| `atmospheric_tension_beds` | 20 | `spectral_wind_tension_bed_v1` | 706.9 | 0.0019 | 0.2878 | 12.66 | 0.0216 |
| `irregular_rhythmic_transients` | 20 | `stochastic_percussive_transient_generator_v1` | 652.0 | 0.0253 | 0.1506 | 12.56 | 0.3963 |
| `synthetic_creature_sounds` | 20 | `formant_vocal_tract_creature_synth_v1` | 976.9 | 0.0035 | 0.0040 | 12.99 | 0.0706 |
| `spectral_textures` | 20 | `additive_microtonal_spectral_bank_v1` | 1,456.3 | 0.0246 | 0.0080 | 10.20 | 0.0373 |
| `unstable_pitch_modulated_tones` | 20 | `chaotic_glissando_pitch_modulator_v1` | 530.0 | 0.0059 | 0.0296 | 7.47 | 0.0583 |

### Provisional label distributions (heuristic output, not human judgments)

- `primary_emotion`: unease 41 · suspense 34 · anxiety 28 · shock 24 · dread 20 · disorientation 20 · anticipation 19 · uncertainty 14
- `intensity`: moderate 99 · high 55 · low 26 · extreme 20 · subtle 0
- `tension`: sustained 87 · building 65 · latent 29 · escalating 19 · climactic 0

## Dataset creation

### Source audio

All 200 waveforms were synthesized by deterministic DSP code (`numpy`, `scipy.signal`) in the source repository
(`src/generate_audio.py`), one engine per category, driven by per-recipe parameters and a recorded `random_seed`.
Rerunning the generator with the committed seed reproduces every file byte-for-byte; the SHA-256 in each row is the
check. No field recordings, sample libraries, or pretrained / neural audio models were used at any stage.

### Annotation process

1. **Measurement** — 12 acoustic features per clip via windowed FFT/STFT and temporal envelope analysis (`src/analyze_audio.py`).
2. **Provisional labeling** — a rule-based heuristic (`src/generate_metadata.py`) mapped synthesis parameters and measured features to the controlled vocabularies, scores, event regions and a templated caption. These are the labels shipped here.
3. **Human review (not yet performed)** — the Label Studio interface in `supplementary/label_studio_annotation_config.xml` lets reviewers listen, adjust regions, correct tags, rewrite captions and set `human_review_status`. As of v0.1, **no record has gone through this step**. There are no annotators to describe, no inter-annotator agreement to report, and no claim of expert validation is made.

### Personal and sensitive information

None. The audio is synthetic; there are no voices, people, places, or recordings of real events.

## Considerations for using the data

- **Provisional labels.** Affective and semantic labels are heuristic proposals. Models trained on them learn the heuristic, not human perception. Treat any downstream metric as a measurement against algorithmic targets until the review pass is complete.
- **Synthetic domain.** Clips are short (2.00–3.25 s), mono, 16 kHz, and come from 10 engines with 10 recipes each. Expect strong within-category acoustic regularities and a large domain gap to produced film audio.
- **Small size.** 200 clips is a schema-validation and pipeline MVP, not a training corpus.
- **Labels follow the recipe.** `sound_event_category` is determined by the synthesis engine, so classification baselines will look optimistic.
- **Captions are templated.** They are grounded in parameters and features, not in listening; lexical diversity is limited.

## Licensing and redistribution status

**No public redistribution licence has been selected for this dataset.** The Hub `license` field is set to
`other` with `license_name: zmh-private-evaluation-no-license-selected` as a deliberate placeholder, and the
repository is private. Access is for internal evaluation by Zazie Productions LLC and parties it explicitly
authorises. Do not redistribute, mirror, or re-license any part of this repository. See `LICENSE.md`.

Rights position in brief (full framework: `supplementary/IP_OWNERSHIP.md`):

- **Zazie Productions LLC — Zazie Media Holdings** controls the compilation, schema/taxonomy, curation, captions and documentation (Categories B and C).
- **Procedural waveforms and unreviewed algorithmic labels** are classified as Category D: purely machine-generated outputs whose standalone copyright status under U.S. law is uncertain. Zazie does not assert standalone sound-recording copyright over them.
- **Acoustic measurements** are factual and treated as noncopyrightable (Category D).
- **No third-party content** is included; the upstream Label Studio platform is not included and remains Apache-2.0 (Category A).

Any change to the licence, visibility, or inclusion of third-party content requires a separate decision by the rights holder.

## Reproducibility and integrity checks

- `MANIFEST.sha256` lists the SHA-256 of every file in this repository; each row's `sha256_checksum` must match its WAV.
- The source repository regenerates the audio (`python3 -m src.pipeline --count 200 --sample-rate 16000 --base-seed 202610`), validates the records against `supplementary/dataset_schema.json`, and rebuilds this folder with `python3 -m src.export_huggingface build`.
- Rows here are a one-to-one, field-preserving projection of the source `training_data.jsonl` plus the per-sample columns of `provenance_manifest.csv`; the only transformations are `audio_filename → file_name` (consumed into `audio`) and `synthesis_parameters.additional_params → additional_params_json`.

## Citation

```bibtex
@misc{zmh_horror_sound_semantics_v01,
  title        = {ZMH Horror Sound Semantics (v0.1): A Provenance-Audited Procedural Horror Audio Dataset with Audio-Language Semantics},
  author       = {{Zazie Productions LLC — Zazie Media Holdings}},
  year         = {2026},
  note         = {Version 0.1 (MVP). 200 procedurally synthesized clips; all annotations algorithmic and unreviewed. Private evaluation release; no public licence selected.},
}
```

The annotation tooling builds on Label Studio (HumanSignal / Heartex, Apache-2.0); please credit upstream when citing the labeling pipeline.
