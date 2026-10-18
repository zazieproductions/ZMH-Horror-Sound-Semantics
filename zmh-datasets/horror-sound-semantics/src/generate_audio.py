"""
Procedural Digital Audio Synthesis Engine for ZMH Horror Sound Semantics (v0.1).

Generates original horror and suspense audio examples using purely mathematical
oscillators, filtered noise, FM/AM/RM modulation, formant filtering, modal resonance,
and nonlinear signal processing.

IP & Provenance Notice:
- Uses 0% copyrighted recordings, 0% third-party samples, and 0% pretrained AI models.
- Outputs are deterministically reproducible from documented synthesis parameters and random seeds.
- Procedurally generated waveforms do not automatically qualify for copyright protection;
  human creative contributions (DSP architecture, parameter curation, schema design) are
  documented separately in `provenance_manifest.csv` and `docs/IP_OWNERSHIP.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

SOUND_EVENT_CATEGORIES: List[str] = [
    "dissonant_drones",
    "metallic_resonance",
    "glitching_machinery",
    "pulsating_bass_textures",
    "digital_interference",
    "atmospheric_tension_beds",
    "irregular_rhythmic_transients",
    "synthetic_creature_sounds",
    "spectral_textures",
    "unstable_pitch_modulated_tones",
]

SPLIT_BY_RECIPE_INDEX: Dict[int, str] = {
    1: "train",
    2: "train",
    3: "train",
    4: "validation",
    5: "test",
}


@dataclass
class GeneratedSampleSpec:
    """Complete specification and provenance record for a single procedural audio sample."""

    sample_id: str
    audio_filename: str
    audio_filepath: str
    sound_event_category: str
    recipe_id: str
    variant_index: int
    split: str
    random_seed: int
    duration: float
    sample_rate: int
    sha256_checksum: str
    synthesis_parameters: Dict[str, Any]
    temporal_events: List[Dict[str, Any]]


def compute_file_sha256(filepath: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _apply_envelope(
    waveform: np.ndarray,
    sr: int,
    attack_sec: float,
    release_sec: float,
) -> np.ndarray:
    """Apply smooth cosine/linear attack and release envelope to prevent boundary clicks."""
    n = len(waveform)
    env = np.ones(n, dtype=np.float64)
    attack_samples = max(1, min(int(attack_sec * sr), n // 2))
    release_samples = max(1, min(int(release_sec * sr), n // 2))

    env[:attack_samples] = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, attack_samples)))
    env[n - release_samples :] = 0.5 * (
        1.0 + np.cos(np.linspace(0.0, math.pi, release_samples))
    )
    return waveform * env


def _simple_schroeder_reverb(
    waveform: np.ndarray,
    sr: int,
    decay_sec: float,
    wet_mix: float = 0.28,
) -> np.ndarray:
    """Lightweight deterministic feedback comb/allpass reverb network."""
    if decay_sec <= 0.02 or wet_mix <= 0.0:
        return waveform
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    wet = np.zeros_like(waveform, dtype=np.float64)
    for delay_ms in comb_delays_ms:
        delay_samples = max(1, int((delay_ms / 1000.0) * sr))
        g = 10.0 ** (-3.0 * (delay_ms / 1000.0) / max(0.05, decay_sec))
        g = min(0.92, max(0.1, g))
        b = np.zeros(delay_samples + 1, dtype=np.float64)
        b[0] = 1.0
        a = np.zeros(delay_samples + 1, dtype=np.float64)
        a[0] = 1.0
        a[-1] = -g
        wet += signal.lfilter(b, a, waveform) * 0.25
    return (1.0 - wet_mix) * waveform + wet_mix * wet


def _biquad_filter(
    waveform: np.ndarray,
    sr: int,
    cutoff_hz: float,
    btype: str = "lowpass",
    bandwidth_hz: float = 250.0,
) -> np.ndarray:
    """Apply stable 2nd-order Butterworth filter."""
    nyq = 0.5 * sr
    if btype in ("lowpass", "highpass"):
        norm_freq = min(0.96, max(0.01, cutoff_hz / nyq))
        sos = signal.butter(2, norm_freq, btype=btype, output="sos")
    elif btype == "bandpass":
        low = max(20.0, cutoff_hz - 0.5 * bandwidth_hz)
        high = min(nyq * 0.95, cutoff_hz + 0.5 * bandwidth_hz)
        if high <= low + 10.0:
            high = min(nyq * 0.96, low + 50.0)
        sos = signal.butter(
            2,
            [max(0.01, low / nyq), min(0.96, high / nyq)],
            btype="bandpass",
            output="sos",
        )
    else:
        return waveform
    return signal.sosfilt(sos, waveform)


def _normalize_to_int16(waveform: np.ndarray, target_peak: float = 0.891) -> np.ndarray:
    """Remove DC offset, normalize peak amplitude to target_peak (-1 dBFS), and quantize to int16."""
    centered = waveform - np.mean(waveform)
    peak = float(np.max(np.abs(centered)))
    if peak < 1e-8:
        centered = np.zeros_like(centered)
    else:
        centered = centered * (target_peak / peak)
    clipped = np.clip(centered, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def _synthesize_category_waveform(
    category: str,
    rng: np.random.Generator,
    sr: int,
    duration: float,
    recipe_idx: int,
    variant_idx: int,
    seed: int,
    recipe_id: str,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """Dispatch synthesis to the category-specific procedural DSP engine."""
    n = int(round(duration * sr))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)

    if category == "dissonant_drones":
        base_freq = float(np.round(48.0 + recipe_idx * 11.5 + variant_idx * 3.25, 2))
        mod_rate = float(np.round(0.22 + recipe_idx * 0.14 + variant_idx * 0.05, 3))
        mod_depth = float(np.round(0.35 + recipe_idx * 0.08, 3))
        cutoff = float(np.round(380.0 + recipe_idx * 140.0 + variant_idx * 45.0, 1))
        drive = float(np.round(1.6 + recipe_idx * 0.35, 2))
        noise_mix = float(np.round(0.06 + variant_idx * 0.02, 3))
        reverb_decay = float(np.round(1.2 + recipe_idx * 0.25, 2))
        attack = float(np.round(0.35 + variant_idx * 0.1, 2))
        release = float(np.round(0.45 + variant_idx * 0.1, 2))

        # Minor second (16/15), tritone (sqrt(2)), and detuned sub-octave
        intervals = [0.5, 1.0, 16.0 / 15.0, math.sqrt(2.0), 1.498]
        sig = np.zeros(n, dtype=np.float64)
        for idx, ratio in enumerate(intervals):
            detune = rng.uniform(-0.8, 0.8)
            freq = base_freq * ratio + detune
            lfo = np.sin(2.0 * math.pi * (mod_rate * (1.0 + 0.15 * idx)) * t)
            phase = 2.0 * math.pi * freq * t + mod_depth * lfo
            osc = 0.65 * np.sin(phase) + 0.35 * signal.sawtooth(phase)
            sig += osc / len(intervals)

        noise = rng.standard_normal(n) * noise_mix
        sig = np.tanh(drive * (sig + noise))
        sig = _biquad_filter(sig, sr, cutoff, btype="lowpass")
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.32)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "event_label": "dissonant_drones",
            }
        ]
        params = {
            "engine_name": "detuned_tritone_cluster_drone_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "lowpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"interval_ratios": [round(x, 4) for x in intervals]},
        }
        return sig, params, events

    if category == "metallic_resonance":
        base_freq = float(np.round(210.0 + recipe_idx * 68.0 + variant_idx * 23.0, 2))
        mod_rate = float(np.round(32.0 + recipe_idx * 14.0 + variant_idx * 5.0, 2))
        mod_depth = float(np.round(0.45 + variant_idx * 0.12, 3))
        cutoff = float(np.round(2600.0 + recipe_idx * 380.0, 1))
        drive = float(np.round(1.3 + recipe_idx * 0.2, 2))
        noise_mix = float(np.round(0.08 + variant_idx * 0.03, 3))
        reverb_decay = float(np.round(1.6 + recipe_idx * 0.3, 2))
        attack = 0.015
        release = float(np.round(0.35 + variant_idx * 0.08, 2))

        # Inharmonic plate/bell partial ratios
        ratios = [1.0, 1.4142, 1.7321, 2.2361, 2.7183, 3.1416, 3.873, 4.5826, 5.3852]
        strike_times = [0.05]
        if recipe_idx % 2 == 0:
            strike_times.append(round(duration * 0.48, 3))

        sig = np.zeros(n, dtype=np.float64)
        events = []
        for s_idx, t0 in enumerate(strike_times):
            t_rel = np.maximum(0.0, t - t0)
            mask = (t >= t0).astype(np.float64)
            strike_sig = np.zeros(n, dtype=np.float64)
            for p_idx, r in enumerate(ratios):
                freq = base_freq * r * (1.0 + 0.02 * s_idx)
                if freq >= 0.46 * sr:
                    continue
                decay_rate = 1.4 + 0.45 * p_idx
                amp = (0.85 ** p_idx) * np.exp(-decay_rate * t_rel) * mask
                strike_sig += amp * np.sin(2.0 * math.pi * freq * t_rel + rng.uniform(0, 2 * math.pi))
            # Add metallic impact burst
            burst_env = np.exp(-38.0 * t_rel) * mask
            strike_sig += noise_mix * burst_env * rng.standard_normal(n)
            sig += strike_sig
            end_ev = strike_times[s_idx + 1] if s_idx + 1 < len(strike_times) else duration
            events.append(
                {
                    "start_sec": round(t0, 3),
                    "end_sec": round(end_ev, 3),
                    "event_label": "metallic_resonance",
                }
            )

        # Ring modulation shimmer
        ring = 1.0 - mod_depth + mod_depth * np.sin(2.0 * math.pi * mod_rate * t)
        sig = np.tanh(drive * sig * ring)
        sig = _biquad_filter(sig, sr, cutoff, btype="lowpass")
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.4)
        sig = _apply_envelope(sig, sr, attack, release)

        params = {
            "engine_name": "inharmonic_modal_plate_resonator_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "lowpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {
                "modal_ratios": ratios,
                "strike_Count": len(strike_times),
            },
        }
        return sig, params, events

    if category == "glitching_machinery":
        base_freq = float(np.round(95.0 + recipe_idx * 28.0 + variant_idx * 11.0, 2))
        mod_rate = float(np.round(7.5 + recipe_idx * 2.4 + variant_idx * 1.1, 2))
        mod_depth = float(np.round(0.65 + variant_idx * 0.1, 3))
        cutoff = float(np.round(1450.0 + recipe_idx * 260.0, 1))
        drive = float(np.round(2.1 + recipe_idx * 0.3, 2))
        noise_mix = float(np.round(0.22 + variant_idx * 0.05, 3))
        reverb_decay = float(np.round(0.35 + recipe_idx * 0.08, 2))
        attack = 0.02
        release = 0.08

        # Stepped sample-and-hold pitch and gate sequence
        step_hz = mod_rate
        step_indices = np.floor(t * step_hz).astype(int)
        num_steps = int(np.max(step_indices)) + 2
        step_pitches = base_freq * rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.66], size=num_steps)
        step_gates = rng.choice([0.15, 1.0, 1.0, 0.05, 1.0], size=num_steps)
        inst_freq = step_pitches[step_indices]
        gate = step_gates[step_indices]

        phase = 2.0 * math.pi * np.cumsum(inst_freq) / sr
        mech_osc = 0.6 * signal.square(phase, duty=0.35) + 0.4 * signal.sawtooth(phase)
        clicks = np.diff(step_indices, prepend=step_indices[0]).astype(np.float64)
        click_burst = signal.lfilter([1.0], [1.0, -0.85], clicks) * 0.8
        raw = gate * (mech_osc + noise_mix * rng.standard_normal(n)) + click_burst

        # Bitcrush quantization effect
        levels = 12.0 + recipe_idx * 4.0
        crushed = np.round(np.tanh(drive * raw) * levels) / levels
        sig = _biquad_filter(crushed, sr, cutoff, btype="bandpass", bandwidth_hz=1200.0)
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.18)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration * 0.5, 3),
                "event_label": "glitching_machinery",
            },
            {
                "start_sec": round(duration * 0.5, 3),
                "end_sec": round(duration, 3),
                "event_label": "glitching_machinery",
            },
        ]
        params = {
            "engine_name": "stepped_buffer_stutter_machinery_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "bandpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"quantization_levels": int(levels)},
        }
        return sig, params, events

    if category == "pulsating_bass_textures":
        base_freq = float(np.round(42.0 + recipe_idx * 7.5 + variant_idx * 2.5, 2))
        mod_rate = float(np.round(1.6 + recipe_idx * 0.55 + variant_idx * 0.25, 2))
        mod_depth = float(np.round(0.75 + variant_idx * 0.08, 3))
        cutoff = float(np.round(190.0 + recipe_idx * 45.0 + variant_idx * 20.0, 1))
        drive = float(np.round(2.4 + recipe_idx * 0.25, 2))
        noise_mix = float(np.round(0.03 + variant_idx * 0.01, 3))
        reverb_decay = float(np.round(0.55 + recipe_idx * 0.1, 2))
        attack = 0.12
        release = 0.25

        # Sub-bass 2-op FM + asymmetric LFO pulse
        lfo = 0.5 * (1.0 + np.sin(2.0 * math.pi * mod_rate * t - math.pi / 2.0)) ** 1.8
        mod_osc = np.sin(2.0 * math.pi * (base_freq * 2.0) * t)
        carrier = np.sin(
            2.0 * math.pi * base_freq * t + (1.2 + 1.8 * lfo) * mod_osc
        )
        sub_osc = np.sin(2.0 * math.pi * (base_freq * 0.5) * t)
        pulse_env = 1.0 - mod_depth + mod_depth * lfo

        raw = pulse_env * (0.7 * carrier + 0.3 * sub_osc + noise_mix * rng.standard_normal(n))
        sig = np.tanh(drive * raw)
        sig = _biquad_filter(sig, sr, cutoff, btype="lowpass")
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.15)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "event_label": "pulsating_bass_textures",
            }
        ]
        params = {
            "engine_name": "sub_bass_fm_throbbing_saturator_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "lowpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"fm_ratio": 2.0},
        }
        return sig, params, events

    if category == "digital_interference":
        base_freq = float(np.round(1350.0 + recipe_idx * 320.0 + variant_idx * 140.0, 2))
        mod_rate = float(np.round(19.0 + recipe_idx * 6.5 + variant_idx * 3.0, 2))
        mod_depth = float(np.round(0.8 + variant_idx * 0.08, 3))
        cutoff = float(np.round(1200.0 + recipe_idx * 250.0, 1))
        drive = float(np.round(2.5 + recipe_idx * 0.3, 2))
        noise_mix = float(np.round(0.38 + variant_idx * 0.06, 3))
        reverb_decay = float(np.round(0.2 + recipe_idx * 0.05, 2))
        attack = 0.01
        release = 0.04

        # Fast frequency-hopping carrier + sample-rate decimation + highpass static bursts
        hop_idx = np.floor(t * mod_rate).astype(int)
        n_hops = int(np.max(hop_idx)) + 2
        hop_mults = rng.choice([0.6, 1.0, 1.414, 1.85, 2.3, 0.0], size=n_hops)
        inst_freq = np.minimum(0.44 * sr, base_freq * hop_mults[hop_idx])
        phase = 2.0 * math.pi * np.cumsum(inst_freq) / sr
        carrier = np.sin(phase) * (hop_mults[hop_idx] > 0).astype(np.float64)

        # Sample-and-hold decimation (aliasing artifact)
        decim = 3 + (recipe_idx % 4)
        aliased = np.repeat(carrier[::decim], decim)[:n]
        if len(aliased) < n:
            aliased = np.pad(aliased, (0, n - len(aliased)))

        static = _biquad_filter(rng.standard_normal(n), sr, cutoff, btype="highpass")
        packet_gate = (rng.uniform(0, 1, size=n_hops)[hop_idx] > 0.2).astype(np.float64)
        sig = packet_gate * ((1.0 - noise_mix) * aliased + noise_mix * static)
        sig = np.tanh(drive * sig)
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.12)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "event_label": "digital_interference",
            }
        ]
        params = {
            "engine_name": "aliased_packet_corrupt_interference_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "highpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"decimation_factor": decim},
        }
        return sig, params, events

    if category == "atmospheric_tension_beds":
        base_freq = float(np.round(110.0 + recipe_idx * 24.0 + variant_idx * 9.0, 2))
        mod_rate = float(np.round(0.18 + recipe_idx * 0.09 + variant_idx * 0.04, 3))
        mod_depth = float(np.round(0.5 + variant_idx * 0.1, 3))
        cutoff = float(np.round(620.0 + recipe_idx * 160.0, 1))
        drive = float(np.round(1.25 + recipe_idx * 0.15, 2))
        noise_mix = float(np.round(0.55 + variant_idx * 0.08, 3))
        reverb_decay = float(np.round(1.8 + recipe_idx * 0.3, 2))
        attack = float(np.round(0.55 + variant_idx * 0.12, 2))
        release = float(np.round(0.4 + variant_idx * 0.1, 2))

        # Pink-like filtered noise + slow sweeping resonant bandpass + eerie minor-ninth pad
        white = rng.standard_normal(n)
        pink = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408], [1.0, -2.494956, 2.017265, -0.522189], white)
        band1 = _biquad_filter(pink, sr, cutoff, btype="bandpass", bandwidth_hz=280.0)
        band2 = _biquad_filter(white, sr, cutoff * 2.1, btype="bandpass", bandwidth_hz=450.0)

        # Eerie minor-9th harmonic shimmer pad with rising crescendo
        crescendo = np.linspace(0.25, 1.0, n) ** 1.4
        pad = (
            0.5 * np.sin(2.0 * math.pi * base_freq * t)
            + 0.3 * np.sin(2.0 * math.pi * (base_freq * 1.5) * t)
            + 0.2 * np.sin(2.0 * math.pi * (base_freq * 2.118) * t + np.sin(2.0 * math.pi * mod_rate * t))
        )
        sig = crescendo * (noise_mix * (0.7 * band1 + 0.3 * band2) + (1.0 - noise_mix) * pad)
        sig = np.tanh(drive * sig)
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.45)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "event_label": "atmospheric_tension_beds",
            }
        ]
        params = {
            "engine_name": "spectral_wind_tension_bed_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "dual_bandpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"crescendo_exponent": 1.4},
        }
        return sig, params, events

    if category == "irregular_rhythmic_transients":
        base_freq = float(np.round(140.0 + recipe_idx * 42.0 + variant_idx * 15.0, 2))
        mod_rate = float(np.round(3.2 + recipe_idx * 0.9 + variant_idx * 0.4, 2))
        mod_depth = float(np.round(0.7 + variant_idx * 0.1, 3))
        cutoff = float(np.round(1800.0 + recipe_idx * 320.0, 1))
        drive = float(np.round(2.0 + recipe_idx * 0.25, 2))
        noise_mix = float(np.round(0.32 + variant_idx * 0.06, 3))
        reverb_decay = float(np.round(0.65 + recipe_idx * 0.12, 2))
        attack = 0.008
        release = 0.12

        # Generate irregular transient strike times across clip
        num_hits = 5 + recipe_idx + variant_idx
        raw_intervals = rng.uniform(0.14, 0.55, size=num_hits)
        hit_times = np.cumsum(raw_intervals)
        hit_times = hit_times / (hit_times[-1] + 0.25) * (duration - 0.25) + 0.06

        sig = np.zeros(n, dtype=np.float64)
        events = []
        for h_idx, ht in enumerate(hit_times):
            t_rel = np.maximum(0.0, t - ht)
            mask = (t >= ht).astype(np.float64)
            hit_amp = rng.uniform(0.45, 1.0)
            pitch_drop = base_freq * (1.0 + 1.8 * np.exp(-35.0 * t_rel))
            phase = 2.0 * math.pi * pitch_drop * t_rel
            tonal_hit = np.exp(-18.0 * t_rel) * np.sin(phase)
            noise_hit = np.exp(-42.0 * t_rel) * rng.standard_normal(n)
            sig += hit_amp * mask * ((1.0 - noise_mix) * tonal_hit + noise_mix * noise_hit)
            events.append(
                {
                    "start_sec": round(float(ht), 3),
                    "end_sec": round(float(min(duration, ht + 0.18)), 3),
                    "event_label": "irregular_rhythmic_transients",
                }
            )

        sig = np.tanh(drive * sig)
        sig = _biquad_filter(sig, sr, cutoff, btype="lowpass")
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.25)
        sig = _apply_envelope(sig, sr, attack, release)

        params = {
            "engine_name": "stochastic_percussive_transient_generator_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "lowpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"transient_hit_count": len(hit_times)},
        }
        return sig, params, events

    if category == "synthetic_creature_sounds":
        base_freq = float(np.round(82.0 + recipe_idx * 26.0 + variant_idx * 12.0, 2))
        mod_rate = float(np.round(21.0 + recipe_idx * 3.5 + variant_idx * 2.0, 2))
        mod_depth = float(np.round(0.68 + variant_idx * 0.1, 3))
        cutoff = float(np.round(1100.0 + recipe_idx * 210.0, 1))
        drive = float(np.round(2.6 + recipe_idx * 0.3, 2))
        noise_mix = float(np.round(0.24 + variant_idx * 0.04, 3))
        reverb_decay = float(np.round(0.85 + recipe_idx * 0.18, 2))
        attack = 0.08
        release = 0.22

        # Nonlinear glottal pitch contour + guttural flutter AM + 3-formant vocal tract filters
        contour = (
            1.0
            + 0.45 * np.sin(2.0 * math.pi * (0.7 / duration) * t)
            - 0.25 * (t / duration) ** 2
            + 0.08 * np.sin(2.0 * math.pi * mod_rate * t)
        )
        inst_freq = base_freq * contour
        phase = 2.0 * math.pi * np.cumsum(inst_freq) / sr
        glottal = 0.65 * signal.sawtooth(phase, width=0.2) + 0.35 * np.sin(phase * 0.5)
        growl_am = 1.0 - mod_depth * 0.5 * (1.0 + np.sin(2.0 * math.pi * mod_rate * t))
        excitation = growl_am * (glottal + noise_mix * rng.standard_normal(n))
        excitation = np.tanh(drive * excitation)

        f1 = _biquad_filter(excitation, sr, 340.0 + recipe_idx * 45.0, btype="bandpass", bandwidth_hz=140.0)
        f2 = _biquad_filter(excitation, sr, cutoff, btype="bandpass", bandwidth_hz=260.0)
        f3 = _biquad_filter(excitation, sr, min(0.42 * sr, cutoff * 2.1), btype="bandpass", bandwidth_hz=380.0)
        sig = 0.5 * f1 + 0.35 * f2 + 0.15 * f3
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.28)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.05,
                "end_sec": round(duration - 0.05, 3),
                "event_label": "synthetic_creature_sounds",
            }
        ]
        params = {
            "engine_name": "formant_vocal_tract_creature_synth_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "triple_formant_bandpass",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {
                "formant_f1_hz": round(340.0 + recipe_idx * 45.0, 1),
                "formant_f2_hz": cutoff,
            },
        }
        return sig, params, events

    if category == "spectral_textures":
        base_freq = float(np.round(260.0 + recipe_idx * 55.0 + variant_idx * 19.0, 2))
        mod_rate = float(np.round(0.45 + recipe_idx * 0.22 + variant_idx * 0.09, 3))
        mod_depth = float(np.round(0.6 + variant_idx * 0.08, 3))
        cutoff = float(np.round(3200.0 + recipe_idx * 350.0, 1))
        drive = float(np.round(1.15 + recipe_idx * 0.1, 2))
        noise_mix = float(np.round(0.05 + variant_idx * 0.02, 3))
        reverb_decay = float(np.round(2.1 + recipe_idx * 0.35, 2))
        attack = float(np.round(0.45 + variant_idx * 0.1, 2))
        release = float(np.round(0.5 + variant_idx * 0.1, 2))

        # 18-partial additive microtonal bank with independent amplitude trajectories
        num_partials = 18
        sig = np.zeros(n, dtype=np.float64)
        for p in range(num_partials):
            ratio = (1.0 + 0.31 * p) + rng.uniform(-0.015, 0.015)
            freq = base_freq * ratio
            if freq >= 0.45 * sr:
                continue
            lfo_freq = mod_rate * (0.5 + 0.18 * p)
            lfo_phase = rng.uniform(0, 2.0 * math.pi)
            amp_env = 0.5 * (1.0 + np.sin(2.0 * math.pi * lfo_freq * t + lfo_phase))
            partial_weight = 1.0 / (1.0 + 0.18 * p)
            sig += partial_weight * amp_env * np.sin(2.0 * math.pi * freq * t + rng.uniform(0, 2.0 * math.pi))

        sig = np.tanh(drive * (sig + noise_mix * rng.standard_normal(n)))
        sig = _biquad_filter(sig, sr, min(0.46 * sr, cutoff), btype="lowpass")
        sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.42)
        sig = _apply_envelope(sig, sr, attack, release)

        events = [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "event_label": "spectral_textures",
            }
        ]
        params = {
            "engine_name": "additive_microtonal_spectral_bank_v1",
            "recipe_id": recipe_id,
            "random_seed": seed,
            "base_frequency_hz": base_freq,
            "modulation_rate_hz": mod_rate,
            "modulation_depth": mod_depth,
            "filter_type": "lowpass_butterworth_2nd",
            "filter_cutoff_hz": cutoff,
            "noise_mix_ratio": noise_mix,
            "nonlinear_drive": drive,
            "reverb_decay_sec": reverb_decay,
            "envelope_attack_sec": attack,
            "envelope_release_sec": release,
            "additional_params": {"partial_count": num_partials},
        }
        return sig, params, events

    # Default / 10th category: "unstable_pitch_modulated_tones"
    base_freq = float(np.round(240.0 + recipe_idx * 48.0 + variant_idx * 22.0, 2))
    mod_rate = float(np.round(4.2 + recipe_idx * 1.3 + variant_idx * 0.6, 2))
    mod_depth = float(np.round(0.55 + variant_idx * 0.12, 3))
    cutoff = float(np.round(2200.0 + recipe_idx * 280.0, 1))
    drive = float(np.round(1.5 + recipe_idx * 0.2, 2))
    noise_mix = float(np.round(0.07 + variant_idx * 0.02, 3))
    reverb_decay = float(np.round(1.1 + recipe_idx * 0.2, 2))
    attack = 0.18
    release = 0.28

    # Dual-LFO + smoothed random-walk pitch instability + downward glissando
    glide = 1.0 - 0.28 * (t / duration)
    lfo1 = np.sin(2.0 * math.pi * mod_rate * t)
    lfo2 = np.sin(2.0 * math.pi * (mod_rate * 0.37) * t + 1.1)
    walk = np.cumsum(rng.standard_normal(n)) / math.sqrt(sr)
    walk = _biquad_filter(walk, sr, 8.0, btype="lowpass")
    inst_freq1 = base_freq * glide * (1.0 + 0.12 * mod_depth * lfo1 + 0.08 * mod_depth * lfo2 + 0.04 * walk)
    inst_freq2 = inst_freq1 * 1.05946  # Minor-second shadow voice

    phase1 = 2.0 * math.pi * np.cumsum(np.maximum(20.0, inst_freq1)) / sr
    phase2 = 2.0 * math.pi * np.cumsum(np.maximum(20.0, inst_freq2)) / sr
    sig = 0.6 * np.sin(phase1) + 0.4 * signal.sawtooth(phase2, width=0.5)
    sig = np.tanh(drive * (sig + noise_mix * rng.standard_normal(n)))
    sig = _biquad_filter(sig, sr, cutoff, btype="lowpass")
    sig = _simple_schroeder_reverb(sig, sr, reverb_decay, wet_mix=0.3)
    sig = _apply_envelope(sig, sr, attack, release)

    events = [
        {
            "start_sec": 0.0,
            "end_sec": round(duration, 3),
            "event_label": "unstable_pitch_modulated_tones",
        }
    ]
    params = {
        "engine_name": "chaotic_glissando_pitch_modulator_v1",
        "recipe_id": recipe_id,
        "random_seed": seed,
        "base_frequency_hz": base_freq,
        "modulation_rate_hz": mod_rate,
        "modulation_depth": mod_depth,
        "filter_type": "lowpass_butterworth_2nd",
        "filter_cutoff_hz": cutoff,
        "noise_mix_ratio": noise_mix,
        "nonlinear_drive": drive,
        "reverb_decay_sec": reverb_decay,
        "envelope_attack_sec": attack,
        "envelope_release_sec": release,
        "additional_params": {"glissando_slope": -0.28, "shadow_interval_ratio": 1.0595},
    }
    return sig, params, events


def write_wav_mono_16bit(filepath: Path, samples_int16: np.ndarray, sample_rate: int) -> None:
    """Write 16-bit PCM mono WAV file using Python standard library `wave`."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())


def generate_single_sample(
    sample_index: int,
    category: str,
    recipe_idx: int,
    variant_idx: int,
    output_dir: Path,
    sample_rate: int = 16000,
    base_seed: int = 202610,
    distribute_splits_across_categories: bool = False,
) -> GeneratedSampleSpec:
    """
    Deterministically synthesize and save a single audio sample, returning its specification.
    """
    sample_id = f"zmh_hss_v01_{sample_index:04d}"
    audio_filename = f"{sample_id}.wav"
    audio_filepath = output_dir / audio_filename
    recipe_id = f"{category}_r{recipe_idx:02d}"

    # Deterministic seed derived from base_seed, category index, recipe_idx, variant_idx
    cat_idx = SOUND_EVENT_CATEGORIES.index(category)
    seed = int(base_seed + cat_idx * 1000 + recipe_idx * 100 + variant_idx)
    rng = np.random.default_rng(seed)

    # Deterministic duration between 2.0s and 3.5s
    durations = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25]
    duration = float(durations[(cat_idx + recipe_idx + variant_idx) % len(durations)])

    # Assign split strictly by recipe_id ((cat_idx, recipe_idx)) so variants of the same recipe
    # never leak across splits, even for small batch counts (<100).
    if distribute_splits_across_categories:
        split_key = ((cat_idx + recipe_idx - 1) % 5) + 1
    else:
        split_key = ((recipe_idx - 1) % 5) + 1
    split = SPLIT_BY_RECIPE_INDEX[split_key]

    waveform, synth_params, temporal_events = _synthesize_category_waveform(
        category=category,
        rng=rng,
        sr=sample_rate,
        duration=duration,
        recipe_idx=recipe_idx,
        variant_idx=variant_idx,
        seed=seed,
        recipe_id=recipe_id,
    )
    pcm_int16 = _normalize_to_int16(waveform)
    write_wav_mono_16bit(audio_filepath, pcm_int16, sample_rate)
    sha256_hex = compute_file_sha256(audio_filepath)

    return GeneratedSampleSpec(
        sample_id=sample_id,
        audio_filename=audio_filename,
        audio_filepath=str(audio_filepath),
        sound_event_category=category,
        recipe_id=recipe_id,
        variant_index=variant_idx,
        split=split,
        random_seed=seed,
        duration=duration,
        sample_rate=sample_rate,
        sha256_checksum=sha256_hex,
        synthesis_parameters=synth_params,
        temporal_events=temporal_events,
    )


def generate_dataset_audio(
    output_dir: Path,
    count: int = 100,
    sample_rate: int = 16000,
    base_seed: int = 202610,
) -> List[GeneratedSampleSpec]:
    """
    Generate a balanced batch of procedural horror audio samples across all 10 categories.

    For `count=100` (default):
      - 10 categories x 5 recipes/category x 2 variants/recipe = 100 samples.
      - Split distribution: 60 train, 20 validation, 20 test (stratified across all 10 categories,
        with 0 recipe overlap across splits).
    """
    if count < 10:
        raise ValueError("count must be at least 10 (at least 1 sample per category).")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: List[GeneratedSampleSpec] = []
    sample_index = 1

    num_categories = len(SOUND_EVENT_CATEGORIES)
    samples_per_category = count // num_categories
    remainder = count % num_categories
    distribute_splits = samples_per_category < 10

    for cat_idx, category in enumerate(SOUND_EVENT_CATEGORIES):
        cat_target = samples_per_category + (1 if cat_idx < remainder else 0)
        # Distribute across recipes (2 variants per recipe by default)
        for item_idx in range(cat_target):
            recipe_idx = (item_idx // 2) + 1
            variant_idx = (item_idx % 2) + 1
            spec = generate_single_sample(
                sample_index=sample_index,
                category=category,
                recipe_idx=recipe_idx,
                variant_idx=variant_idx,
                output_dir=output_dir,
                sample_rate=sample_rate,
                base_seed=base_seed,
                distribute_splits_across_categories=distribute_splits,
            )
            specs.append(spec)
            sample_index += 1

    logger.info(
        "Generated %d procedural audio samples in %s (sample_rate=%d Hz)",
        len(specs),
        output_dir,
        sample_rate,
    )
    return specs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate procedural horror audio samples for ZMH Horror Sound Semantics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "demo_audio",
        help="Directory to write generated .wav files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of audio samples to generate (default: 100)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[16000, 22050, 44100, 48000],
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=202610,
        help="Deterministic base random seed",
    )
    args = parser.parse_args()
    specs = generate_dataset_audio(
        output_dir=args.output_dir,
        count=args.count,
        sample_rate=args.sample_rate,
        base_seed=args.base_seed,
    )
    print(f"Successfully generated {len(specs)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
