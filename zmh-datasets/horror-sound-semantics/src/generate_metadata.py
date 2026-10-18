"""
Metadata & Provisional Semantic Label Generator for ZMH Horror Sound Semantics (v0.1).

Combines documented procedural DSP synthesis parameters with measured acoustic signal
features to propose structured semantic annotations and candidate natural-language captions.

Integrity & Provenance Rules:
- Never fabricates human listening judgments or expert review status.
- All algorithmically generated labels are marked with:
    `annotation_provenance = "algorithmic_dsp_and_acoustic_heuristic_v0.1"`
    `human_review_status = "unreviewed_provisional"`
- Psychological/affective categories (`primary_emotion`, `secondary_emotions`, `tension`)
  carry an explicit `psychological_label_disclaimer`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .analyze_audio import AudioAnalysisResult, analyze_wav_file
from .generate_audio import GeneratedSampleSpec

logger = logging.getLogger(__name__)

DATASET_NAME = "ZMH Horror Sound Semantics"
DATASET_VERSION = "0.1"
PUBLISHER = "Zazie Productions LLC — Zazie Media Holdings"

PSYCHOLOGICAL_DISCLAIMER = (
    "Provisional affective labels (primary_emotion, secondary_emotions, tension) are "
    "algorithmically derived from DSP synthesis parameters and measured acoustic signal "
    "features; they do not represent verified human listening judgments until "
    "human_review_status is human_reviewed_verified or human_reviewed_corrected."
)

CATEGORY_SEMANTIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "dissonant_drones": {
        "acoustic_texture": "dense_sub_harmonic",
        "primary_emotions": ["dread", "unease", "suspense"],
        "secondary_pool": ["dread", "unease", "uncertainty", "anxiety", "anticipation"],
        "rhythmic_characteristics": "arrhythmic_sustained",
        "tonality": "dissonant_bichord",
        "spatial_options": ["subterranean_rumble", "cavernous_reverberant", "claustrophobic_intimate"],
        "cinematic_usage": ["slow_burn_dread_underscore", "liminal_space_drone", "ritual_occult_ambience"],
        "timbre_phrase": "detuned tritone and minor-second oscillator cluster drone",
    },
    "metallic_resonance": {
        "acoustic_texture": "inharmonic_metallic",
        "primary_emotions": ["unease", "shock", "suspense"],
        "secondary_pool": ["unease", "dread", "disorientation", "uncertainty", "shock"],
        "rhythmic_characteristics": "stochastic_intermittent",
        "tonality": "inharmonic_partial_series",
        "spatial_options": ["cavernous_reverberant", "focused_foreground_transient", "Industrial_corridor"],
        "cinematic_usage": ["industrial_sci_fi_corridor", "supernatural_manifestation", "jump_scare_stinger"],
        "timbre_phrase": "struck inharmonic modal plate and ring-modulated metallic resonator",
    },
    "glitching_machinery": {
        "acoustic_texture": "granular_stutter",
        "primary_emotions": ["anxiety", "disorientation", "unease"],
        "secondary_pool": ["anxiety", "disorientation", "shock", "uncertainty", "suspense"],
        "rhythmic_characteristics": "fast_mechanical_stutter",
        "tonality": "atonal_noise",
        "spatial_options": ["claustrophobic_intimate", "focused_foreground_transient", "distant_muffled"],
        "cinematic_usage": ["industrial_sci_fi_corridor", "found_footage_signal_corrupt", "chase_sequence_pulse"],
        "timbre_phrase": "stepped sample-and-hold mechanical pulse with bitcrushed buffer stutter",
    },
    "pulsating_bass_textures": {
        "acoustic_texture": "throbbing_low_end",
        "primary_emotions": ["anticipation", "dread", "suspense"],
        "secondary_pool": ["anticipation", "dread", "anxiety", "suspense", "unease"],
        "rhythmic_characteristics": "slow_periodic_pulse",
        "tonality": "dark_pedal_drone",
        "spatial_options": ["subterranean_rumble", "claustrophobic_intimate", "cavernous_reverberant"],
        "cinematic_usage": ["chase_sequence_pulse", "slow_burn_dread_underscore", "psychological_breakdown"],
        "timbre_phrase": "sub-bass two-operator FM pulse with asymmetric LFO waveshaping",
    },
    "digital_interference": {
        "acoustic_texture": "bitcrushed_aliased",
        "primary_emotions": ["disorientation", "shock", "anxiety"],
        "secondary_pool": ["disorientation", "anxiety", "uncertainty", "shock", "unease"],
        "rhythmic_characteristics": "chaotic_polyrhythmic",
        "tonality": "atonal_noise",
        "spatial_options": ["focused_foreground_transient", "claustrophobic_intimate", "disembodied_floating"],
        "cinematic_usage": ["found_footage_signal_corrupt", "psychological_breakdown", "jump_scare_stinger"],
        "timbre_phrase": "aliased frequency-hopping carrier with decimated packet-loss static bursts",
    },
    "atmospheric_tension_beds": {
        "acoustic_texture": "filtered_noise_wash",
        "primary_emotions": ["suspense", "uncertainty", "anticipation"],
        "secondary_pool": ["suspense", "uncertainty", "dread", "unease", "anticipation"],
        "rhythmic_characteristics": "arrhythmic_sustained",
        "tonality": "ambiguous_spectral",
        "spatial_options": ["wide_diffused_field", "cavernous_reverberant", "distant_muffled"],
        "cinematic_usage": ["slow_burn_dread_underscore", "liminal_space_drone", "supernatural_manifestation"],
        "timbre_phrase": "dual-bandpass filtered noise wash layered over a rising minor-ninth tension pad",
    },
    "irregular_rhythmic_transients": {
        "acoustic_texture": "percussive_erratic",
        "primary_emotions": ["shock", "anxiety", "anticipation"],
        "secondary_pool": ["shock", "anxiety", "suspense", "uncertainty", "dread"],
        "rhythmic_characteristics": "stochastic_intermittent",
        "tonality": "atonal_noise",
        "spatial_options": ["focused_foreground_transient", "cavernous_reverberant", "claustrophobic_intimate"],
        "cinematic_usage": ["jump_scare_stinger", "chase_sequence_pulse", "psychological_breakdown"],
        "timbre_phrase": "stochastic percussive transient impacts with damped pitch-drop resonance",
    },
    "synthetic_creature_sounds": {
        "acoustic_texture": "formant_guttural",
        "primary_emotions": ["dread", "shock", "unease"],
        "secondary_pool": ["dread", "shock", "anxiety", "unease", "suspense"],
        "rhythmic_characteristics": "accelerating_tremolo",
        "tonality": "unstable_glissando",
        "spatial_options": ["claustrophobic_intimate", "cavernous_reverberant", "focused_foreground_transient"],
        "cinematic_usage": ["monster_vocal_presence", "jump_scare_stinger", "supernatural_manifestation"],
        "timbre_phrase": "triple-formant vocal-tract synthesis with guttural flutter modulation and nonlinear growl",
    },
    "spectral_textures": {
        "acoustic_texture": "crystalline_spectral",
        "primary_emotions": ["uncertainty", "unease", "suspense"],
        "secondary_pool": ["uncertainty", "unease", "disorientation", "suspense", "anticipation"],
        "rhythmic_characteristics": "arrhythmic_sustained",
        "tonality": "microtonal_cluster",
        "spatial_options": ["disembodied_floating", "wide_diffused_field", "cavernous_reverberant"],
        "cinematic_usage": ["supernatural_manifestation", "ritual_occult_ambience", "liminal_space_drone"],
        "timbre_phrase": "eighteen-partial additive microtonal shimmer bank with evolving phase trajectories",
    },
    "unstable_pitch_modulated_tones": {
        "acoustic_texture": "warped_microtonal",
        "primary_emotions": ["disorientation", "unease", "anxiety"],
        "secondary_pool": ["disorientation", "unease", "uncertainty", "anxiety", "dread"],
        "rhythmic_characteristics": "syncopated_heartbeat",
        "tonality": "unstable_glissando",
        "spatial_options": ["disembodied_floating", "claustrophobic_intimate", "wide_diffused_field"],
        "cinematic_usage": ["psychological_breakdown", "supernatural_manifestation", "liminal_space_drone"],
        "timbre_phrase": "downward glissando tone cluster with dual-LFO and random-walk pitch instability",
    },
}


def _score_to_intensity_label(score: float) -> str:
    if score < 0.25:
        return "subtle"
    if score < 0.45:
        return "low"
    if score < 0.68:
        return "moderate"
    if score < 0.85:
        return "high"
    return "extreme"


def _score_to_tension_label(score: float) -> str:
    if score < 0.25:
        return "latent"
    if score < 0.45:
        return "building"
    if score < 0.68:
        return "sustained"
    if score < 0.85:
        return "escalating"
    return "climactic"


def _derive_spatial_property(
    category: str,
    reverb_decay: float,
    centroid_hz: float,
    low_ratio: float,
) -> str:
    if low_ratio > 0.62 and centroid_hz < 450.0:
        return "subterranean_rumble"
    if reverb_decay >= 1.75:
        if centroid_hz > 1400.0:
            return "disembodied_floating"
        return "cavernous_reverberant"
    if category == "atmospheric_tension_beds":
        return "wide_diffused_field"
    if category in ("irregular_rhythmic_transients", "digital_interference"):
        return "focused_foreground_transient"
    if centroid_hz < 550.0 and reverb_decay < 0.9:
        return "distant_muffled"
    return "claustrophobic_intimate"


def build_candidate_caption(
    spec: GeneratedSampleSpec,
    features: Dict[str, float],
    primary_emotion: str,
    secondary_emotions: List[str],
    intensity: str,
    tension: str,
    spatial: str,
    cinematic_usage: List[str],
) -> str:
    """
    Construct a factual, synthesis- and feature-grounded candidate caption.
    """
    profile = CATEGORY_SEMANTIC_PROFILES[spec.sound_event_category]
    synth = spec.synthesis_parameters
    centroid = features["spectral_centroid_hz"]
    rms = features["rms_energy"]
    crest = features["crest_factor_db"]
    base_hz = synth["base_frequency_hz"]
    mod_hz = synth["modulation_rate_hz"]
    cutoff_hz = synth["filter_cutoff_hz"]

    sec_str = ", ".join(secondary_emotions)
    use_str = ", ".join(u.replace("_", " ") for u in cinematic_usage)
    spatial_str = spatial.replace("_", " ")

    caption = (
        f"[Provisional Algorithmic Caption] A {intensity}-intensity, {tension}-tension "
        f"{profile['timbre_phrase']} ({spec.duration:.2f}s at {spec.sample_rate} Hz) "
        f"centered near {base_hz:.1f} Hz with {mod_hz:.2f} Hz modulation and "
        f"{cutoff_hz:.0f} Hz filter cutoff. Measured acoustic profile shows a "
        f"spectral centroid of {centroid:.0f} Hz, RMS energy of {rms:.3f}, and "
        f"crest factor of {crest:.1f} dB, projecting a {spatial_str} spatial impression "
        f"associated with candidate affective tags '{primary_emotion}' ({sec_str}) "
        f"for {use_str} cues."
    )
    return caption


def generate_record_from_spec_and_analysis(
    spec: GeneratedSampleSpec,
    analysis: AudioAnalysisResult,
) -> Dict[str, Any]:
    """
    Combine a GeneratedSampleSpec and AudioAnalysisResult into a schema-compliant dataset record.
    """
    profile = CATEGORY_SEMANTIC_PROFILES[spec.sound_event_category]
    features = analysis.acoustic_features
    synth = spec.synthesis_parameters

    # Calculate deterministic intensity_score in [0.0, 1.0] from RMS, drive, and spectral centroid
    rms_norm = min(1.0, features["rms_energy"] / 0.42)
    drive_norm = min(1.0, (synth["nonlinear_drive"] - 1.0) / 2.2)
    high_norm = min(1.0, features["spectral_centroid_hz"] / 2800.0)
    intensity_score = round(min(0.98, max(0.12, 0.45 * rms_norm + 0.35 * drive_norm + 0.20 * high_norm)), 4)
    intensity_label = _score_to_intensity_label(intensity_score)

    # Calculate deterministic tension_score in [0.0, 1.0] from modulation rate, onset std, and temporal centroid
    mod_norm = min(1.0, synth["modulation_rate_hz"] / 22.0)
    onset_norm = min(1.0, features["onset_strength_std"] / 0.35)
    temp_norm = min(1.0, abs(features["temporal_centroid_norm"] - 0.5) * 2.5 + 0.45)
    tension_score = round(
        min(0.98, max(0.15, 0.35 * mod_norm + 0.30 * onset_norm + 0.35 * temp_norm)),
        4,
    )
    tension_label = _score_to_tension_label(tension_score)

    # Select primary & secondary emotions deterministically from recipe & variant
    prim_list = profile["primary_emotions"]
    primary_emotion = prim_list[(spec.random_seed + spec.variant_index) % len(prim_list)]

    sec_pool = [e for e in profile["secondary_pool"] if e != primary_emotion]
    sec_count = 2
    start_idx = spec.random_seed % len(sec_pool)
    secondary_emotions = [sec_pool[(start_idx + i) % len(sec_pool)] for i in range(sec_count)]

    spatial = _derive_spatial_property(
        category=spec.sound_event_category,
        reverb_decay=synth["reverb_decay_sec"],
        centroid_hz=features["spectral_centroid_hz"],
        low_ratio=features["low_frequency_energy_ratio"],
    )

    usage_pool = profile["cinematic_usage"]
    cinematic_usage = [
        usage_pool[spec.variant_index % len(usage_pool)],
        usage_pool[(spec.variant_index + 1) % len(usage_pool)],
    ]
    # Ensure uniqueness while preserving order
    cinematic_usage = list(dict.fromkeys(cinematic_usage))

    caption = build_candidate_caption(
        spec=spec,
        features=features,
        primary_emotion=primary_emotion,
        secondary_emotions=secondary_emotions,
        intensity=intensity_label,
        tension=tension_label,
        spatial=spatial,
        cinematic_usage=cinematic_usage,
    )

    # Provisional confidence score bounded conservatively [0.62, 0.86] since unreviewed by human
    confidence = round(0.66 + 0.16 * (1.0 - abs(0.5 - features["spectral_flatness"])), 3)

    record: Dict[str, Any] = {
        "sample_id": spec.sample_id,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "publisher": PUBLISHER,
        "audio_filename": spec.audio_filename,
        "duration": spec.duration,
        "sample_rate": analysis.sample_rate,
        "primary_emotion": primary_emotion,
        "secondary_emotions": secondary_emotions,
        "acoustic_texture": profile["acoustic_texture"],
        "sound_event_category": spec.sound_event_category,
        "intensity": intensity_label,
        "intensity_score": intensity_score,
        "tension": tension_label,
        "tension_score": tension_score,
        "rhythmic_characteristics": profile["rhythmic_characteristics"],
        "tonality": profile["tonality"],
        "perceived_spatial_properties": spatial,
        "cinematic_usage": cinematic_usage,
        "natural_language_caption": caption,
        "annotation_confidence": confidence,
        "annotation_provenance": "algorithmic_dsp_and_acoustic_heuristic_v0.1",
        "human_review_status": "unreviewed_provisional",
        "copyright_and_licensing_status": "procedural_audio_unverified_copyright_human_schema_zazie_owned",
        "psychological_label_disclaimer": PSYCHOLOGICAL_DISCLAIMER,
        "split": spec.split,
        "recipe_id": spec.recipe_id,
        "variant_index": spec.variant_index,
        "random_seed": spec.random_seed,
        "generation_method": "procedural_dsp_synthesis",
        "uses_pretrained_ai_model": False,
        "uses_third_party_samples": False,
        "is_authentic_human_or_field_recording": False,
        "sha256_checksum": spec.sha256_checksum,
        "synthesis_parameters": spec.synthesis_parameters,
        "acoustic_features": features,
        "temporal_events": spec.temporal_events,
    }
    return record


def generate_dataset_metadata(
    specs: List[GeneratedSampleSpec],
    existing_records: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate metadata records for a list of GeneratedSampleSpecs.
    Deduplicates by `sample_id` and `sha256_checksum` and preserves any existing
    human-reviewed annotations if re-run idempotently.
    """
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    if existing_records:
        for rec in existing_records:
            sid = rec.get("sample_id")
            if sid:
                existing_by_id[sid] = rec

    records_by_id: Dict[str, Dict[str, Any]] = {}
    seen_hashes: Dict[str, str] = {}

    for spec in specs:
        if spec.sha256_checksum in seen_hashes and seen_hashes[spec.sha256_checksum] != spec.sample_id:
            raise ValueError(
                f"Duplicate audio SHA-256 detected between {spec.sample_id} and "
                f"{seen_hashes[spec.sha256_checksum]}: {spec.sha256_checksum}"
            )
        seen_hashes[spec.sha256_checksum] = spec.sample_id

        # If an existing record was already human-reviewed and has identical audio hash, preserve it
        prev = existing_by_id.get(spec.sample_id)
        if (
            prev is not None
            and prev.get("sha256_checksum") == spec.sha256_checksum
            and prev.get("human_review_status") in ("human_reviewed_verified", "human_reviewed_corrected")
        ):
            records_by_id[spec.sample_id] = prev
            continue

        analysis = analyze_wav_file(spec.audio_filepath)
        record = generate_record_from_spec_and_analysis(spec, analysis)
        records_by_id[spec.sample_id] = record

    sorted_records = [records_by_id[k] for k in sorted(records_by_id.keys())]
    logger.info("Generated %d unique metadata records.", len(sorted_records))
    return sorted_records
