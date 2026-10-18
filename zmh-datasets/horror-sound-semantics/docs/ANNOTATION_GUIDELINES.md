# Annotation Guidelines: ZMH Horror Sound Semantics (v0.1)

**Organization:** Zazie Productions LLC — Zazie Media Holdings  
**Interface Configuration:** `configs/annotation_config.xml`  
**Schema Specification:** `configs/dataset_schema.json`

---

## 1. Purpose & Core Integrity Principles

These guidelines govern human review, correction, and expansion of **ZMH Horror Sound Semantics** inside Label Studio.

### Core Integrity Rules
1. **Never Confuse Algorithmic Pre-Labels with Human Judgments:**
   - Every task imported from `data/label_studio_tasks.json` is pre-populated with a provisional prediction (`model_version: "zmh-hss-provisional-v0.1"`) where `human_review_status` is set to `unreviewed_provisional`.
   - Do **not** mark a sample as `human_reviewed_verified` or `human_reviewed_corrected` without listening to the entire audio clip from start to finish on calibrated headphones or studio monitors.
2. **Accurate Provenance Attribution:**
   - If you listen to the clip and confirm that all provisional semantic choices and the caption are accurate without changes, submit the annotation (or set `human_review_status` to `human_reviewed_verified` and `annotation_provenance` to `human_expert_reviewed_v0.1`).
   - If you change **any** semantic label, temporal boundary, or caption text, set `human_review_status` to `human_reviewed_corrected` and `annotation_provenance` to `human_expert_corrected_v0.1`. (Note: `src/label_studio_integration.py` also detects modifications automatically during ingestion.)
3. **Never Misrepresent Procedural Audio as Authentic Field or Foley Recordings:**
   - Clips with `generation_method = "procedural_dsp_synthesis"` must retain `copyright_and_licensing_status = "procedural_audio_unverified_copyright_human_schema_zazie_owned"`.
   - Only original human-performed instrument or Foley recordings verified by Zazie Productions LLC may be marked `"zazie_verified_original_recording_commercial_licensable"`.

---

## 2. Step-by-Step Labeling Workflow in Label Studio

### Step 1: Waveform Playback & Temporal Segmentation (`temporal_events`)
- Listen to the full clip at comfortable monitoring level.
- Inspect the pre-populated region tag(s) on the `<Audio>` waveform.
- For continuous textures (`dissonant_drones`, `atmospheric_tension_beds`, `spectral_textures`), the region should span the active signal envelope.
- For multi-hit or transient clips (`irregular_rhythmic_transients`, `metallic_resonance`), adjust or add `<Labels>` spans around distinct percussive strikes or resonant events.

### Step 2: Sound-Event Category & Acoustic Texture
Select exactly one **`sound_event_category`**:
- `dissonant_drones` — Sustained low/mid oscillator clusters with minor-second or tritone beating.
- `metallic_resonance` — Inharmonic bell, plate, pipe, or metallic struck resonance.
- `glitching_machinery` — Mechanical stutter, buffer repeat, stepped pitch/gate machinery.
- `pulsating_bass_textures` — Periodic LFO-driven sub-bass or low-frequency FM throbbing.
- `digital_interference` — High-frequency aliasing, packet loss, static bursts, data corruption artifacts.
- `atmospheric_tension_beds` — Filtered noise washes and slowly evolving harmonic pads.
- `irregular_rhythmic_transients` — Erratic, non-isochronous impacts, knocks, or startle hits.
- `synthetic_creature_sounds` — Vocal-tract formant growls, shrieks, or non-human guttural articulations.
- `spectral_textures` — Shimmering additive partial banks and microtonal crystalline clusters.
- `unstable_pitch_modulated_tones` — Continuous glissandi, warped tape-wow, or chaotic vibrato tones.

Select the primary **`acoustic_texture`** that best characterizes the surface timbre (`dense_sub_harmonic`, `inharmonic_metallic`, `granular_stutter`, `throbbing_low_end`, `bitcrushed_aliased`, `filtered_noise_wash`, `percussive_erratic`, `formant_guttural`, `crystalline_spectral`, `warped_microtonal`).

### Step 3: Affective / Emotional Semantics (`primary_emotion`, `secondary_emotions`)
Evaluate the **intended cinematic mood** evoked by the sound cue:
- `dread` — Heavy, ominous, inescapable foreboding (often low-register or dense dissonance).
- `anxiety` — Agitated, restless, accelerating or jittery nervous energy.
- `uncertainty` — Ambiguous, eerie, unresolved tonal or spectral drift.
- `suspense` — Prolonged expectation of an imminent event; rising or sustained tension.
- `shock` — Sudden, high-crest-factor startle or abrasive sensory impact.
- `unease` — Subtle wrongness, microtonal detuning, uncanny resonance.
- `anticipation` — Rhythmic or pulsing buildup drawing attention toward a climax.
- `disorientation` — Sensory confusion caused by erratic pitch glides, aliasing, or chaotic rhythm.

**Rule:** `primary_emotion` (single choice) must **not** be repeated inside `secondary_emotions` (1 to 3 choices).

### Step 4: Intensity, Tension, Rhythm, Tonality & Spatial Impression
- **`intensity`** (`subtle`, `low`, `moderate`, `high`, `extreme`): Perceived loudness, spectral density, and harmonic saturation.
- **`tension`** (`latent`, `building`, `sustained`, `escalating`, `climactic`): Perceived dramatic trajectory across the duration of the clip.
- **`rhythmic_characteristics`**: Choose between `arrhythmic_sustained`, `slow_periodic_pulse`, `fast_mechanical_stutter`, `syncopated_heartbeat`, `stochastic_intermittent`, `accelerating_tremolo`, or `chaotic_polyrhythmic`.
- **`tonality`**: Choose between `atonal_noise`, `microtonal_cluster`, `dissonant_bichord`, `inharmonic_partial_series`, `dark_pedal_drone`, `unstable_glissando`, or `ambiguous_spectral`.
- **`perceived_spatial_properties`**: Choose the dominant reverberant/proximity impression (`claustrophobic_intimate`, `cavernous_reverberant`, `distant_muffled`, `wide_diffused_field`, `focused_foreground_transient`, `subterranean_rumble`, `disembodied_floating`).

### Step 5: Cinematic Usage & Natural-Language Caption
- Select 1 to 4 **`cinematic_usage`** tags describing how a film or game sound designer would deploy the cue.
- Edit **`natural_language_caption`**:
  - Remove the `[Provisional Algorithmic Caption]` prefix once you have human-audited and rewritten or verified the description.
  - Include: (a) physical/timbral description, (b) temporal/rhythmic behavior, (c) spatial character, and (d) dramatic/cinematic function.
  - Keep captions factual, vivid, and between 40 and 1,000 characters.
