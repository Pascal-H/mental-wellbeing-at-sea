#!/usr/bin/env python3
"""
Generate synthetic data for the active speech analysis pipeline.

Creates:
  1. Synthetic metadata CSV with similar structure to the real dataset
  2. Synthetic WAV audio files (noise + sine tones, NOT speech)
  3. Synthetic experiment configuration YAML pointing to the generated data

Privacy notice:
  - All participant IDs are randomly generated UUIDs
  - All label values are drawn from uniform distributions (not matching real distributions)
  - Audio files are purely synthetic (white noise + sine waves), containing no speech
  - Date ranges are shifted and randomised
  - No real data can be reconstructed from these synthetic samples

Usage:
    python generate_synthetic_data.py [--output-dir OUTPUT_DIR] [--seed SEED]
                                       [--n-participants N] [--audio-sr SR]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:
    sf = None
    warnings.warn(
        "soundfile not installed. Audio files will not be generated. "
        "Install with: pip install soundfile"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMPTS = [
    "work_tasks_4718",
    "sustained_utterance_7351",
    "emotion_acting_5839",
    "emotion_acting_4373",
    "count_9822",
]

PRACTICE_PROMPTS = [
    "sustained_utterance_0113",
    "emotion_acting_3130",
    "emotion_acting_5177",
    "emotion_acting_2017",
]

SURVEYS = ["baseline", "daily", "weekly_01", "weekly_02", "final"]


def _seeded_uuid(rng):
    """Generate a deterministic UUID-like string from a seeded RNG."""
    raw = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
    return "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}".format(
        int.from_bytes(raw[0:4], "big"),
        int.from_bytes(raw[4:6], "big"),
        int.from_bytes(raw[6:8], "big"),
        int.from_bytes(raw[8:10], "big"),
        int.from_bytes(raw[10:16], "big"),
    )


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------
def generate_synthetic_audio(duration_s, sr=16_000, rng=None):
    """
    Generate a synthetic audio waveform (noise + sine tones).

    This is NOT speech  -  it is a random signal designed only to be a
    structurally valid WAV file that the pipeline can load and process.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    sr : int
        Sampling rate.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        1-D float32 waveform in [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # White noise component (low amplitude)
    noise = rng.normal(0, 0.05, n_samples).astype(np.float32)

    # Random sine tones (1-3 tones at different frequencies)
    n_tones = rng.integers(1, 4)
    signal = noise.copy()
    for _ in range(n_tones):
        freq = rng.uniform(100, 4000)  # Hz
        amplitude = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        signal += (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(
            np.float32
        )

    # Normalize to [-0.9, 0.9] to avoid clipping
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * 0.9

    return signal


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------
def generate_synthetic_metadata(
    n_participants=15,
    sessions_range=(3, 12),
    audio_dir="",
    rng=None,
):
    """
    Generate a synthetic metadata DataFrame mimicking the real CSV structure.

    Parameters
    ----------
    n_participants : int
        Number of synthetic participants.
    sessions_range : tuple
        (min, max) sessions per participant.
    audio_dir : str
        Base directory where audio files will be (for path construction).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with columns matching the pipeline's expectations.
    list
        List of (relative_wav_path, duration_s) tuples for audio generation.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []
    audio_specs = []  # (relative_path, duration_s)

    # Generate deterministic participant codes (seeded UUID-like strings)
    participant_codes = [_seeded_uuid(rng) for _ in range(n_participants)]
    # Speaker files are separate identifiers (deterministic)
    speaker_files = [_seeded_uuid(rng) for _ in range(n_participants)]

    # Random demographics (completely synthetic, not matching real distributions)
    ages = rng.integers(22, 62, size=n_participants)
    sexes = rng.choice(["male", "female"], size=n_participants)

    # Base date range: synthetic period (Jan-Mar 2024, shifted from real data)
    base_date = pd.Timestamp("2024-01-15")

    for p_idx in range(n_participants):
        n_sessions = rng.integers(sessions_range[0], sessions_range[1] + 1)
        session_dates = sorted(
            [base_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, 60, size=n_sessions)]
        )

        for s_idx, session_date in enumerate(session_dates):
            session_id = f"session_{p_idx:03d}_{s_idx:03d}"

            # Pick a survey type for this session
            survey = rng.choice(SURVEYS, p=[0.05, 0.60, 0.15, 0.15, 0.05])

            # Generate session-level target variables (same for all files in a session,
            # matching the real data where survey scores are per-session)
            session_targets = {
                "phq_8_total_score": float(rng.integers(0, 25)),
                "stress_current": float(rng.uniform(0, 100)),
                "stress_work_tasks": float(rng.uniform(0, 100)),
                "pss_10_total_score": float(rng.integers(0, 41)),
                "who_5_percentage_score_corrected": float(rng.uniform(0, 100)),
                "pss_10_category": rng.choice(["Low", "Moderate", "High"]),
                "who_5_category": rng.choice(["Good", "Poor"]),
                "emotions_current_happy": float(rng.uniform(0, 100)),
                "emotions_current_sad": float(rng.uniform(0, 100)),
                "emotions_current_anger": float(rng.uniform(0, 100)),
                "emotions_current_fear": float(rng.uniform(0, 100)),
                "emotions_current_indifference": float(rng.uniform(0, 100)),
                "emotions_current_shame": float(rng.uniform(0, 100)),
            }

            # Generate files for each prompt in this session
            prompts_this_session = list(PROMPTS)
            # Occasionally add a practice prompt (which will be filtered out)
            if rng.random() < 0.2:
                prompts_this_session.append(rng.choice(PRACTICE_PROMPTS))

            for prompt in prompts_this_session:
                file_uuid = _seeded_uuid(rng)
                # Relative path: speaker_file/session_id/uuid.wav
                rel_path = os.path.join(
                    speaker_files[p_idx], session_id, f"{file_uuid}.wav"
                )

                # Random duration between 2 and 18 seconds
                duration_s = float(rng.uniform(2.0, 18.0))
                audio_specs.append((rel_path, duration_s))

                # All files in a session share the same target values
                row = {
                    "file": rel_path,
                    "participant_code": participant_codes[p_idx],
                    "session": session_id,
                    "survey": survey,
                    "prompt": prompt,
                    "age": int(ages[p_idx]),
                    "sex": sexes[p_idx],
                    "speaker_file": speaker_files[p_idx],
                    "date_file": session_date.strftime("%Y-%m-%d"),
                    "date_survey": session_date.strftime("%Y-%m-%d"),
                }
                row.update(session_targets)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df, audio_specs


# ---------------------------------------------------------------------------
# Experiment config generation
# ---------------------------------------------------------------------------
def generate_experiment_config(
    output_dir,
    csv_path,
    audio_dir,
    test_speakers,
    config_name="synthetic-eGeMAPSv02.yaml",
):
    """
    Generate a synthetic experiment configuration YAML.

    Parameters
    ----------
    output_dir : str
        Directory to write the config file.
    csv_path : str
        Absolute path to the synthetic metadata CSV.
    audio_dir : str
        Absolute path to the synthetic audio directory.
    test_speakers : list
        List of participant codes to use as fixed test speakers.
    config_name : str
        Name of the output config file.

    Returns
    -------
    str
        Path to the generated config file.
    """
    # Use absolute paths for the config
    csv_path_abs = os.path.abspath(csv_path)
    audio_dir_abs = os.path.abspath(audio_dir)

    config_content = f'''# ============================================================================
# Synthetic experiment configuration for the active speech pipeline
# Generated by generate_synthetic_data.py
#
# This configuration uses synthetic data (random noise audio + random labels)
# for pipeline verification. Results have NO scientific meaning.
# ============================================================================

database:
  type: "local"
  path_df_meta: "{csv_path_abs}"
  index_column: ["file"]
  path_data: "{audio_dir_abs}"
  min_sessions: 1
  discard_prompts: ["sustained_utterance_0113", "emotion_acting_3130", "emotion_acting_5177", "emotion_acting_2017"]
  filter_files:
    path_blacklist: null
    filter_lists: []

paths:
  cache_features: "data/synthetic/cache/features"
  cache_vad: "data/synthetic/cache/vad"
  cache_split: "data/synthetic/cache/splitting"
  results_modelling: "results/synthetic/modelling"

AudioProcessor:
  meta:
    num_workers: -1
  preprocessing:
    denoising:
      no_denoising: null
    loudness_normalization:
      no_loudness_normalization: null
  vad:
    no_vad: null
    # To use devaice VAD instead (requires devaice SDK):
    # devaice_vad:
    #   min_segment_length: 0.76
    #   max_segment_length: 6.0
    #   segment_start_delay: 0.150
    #   segment_end_delay: 0.25

FeatureExtractor:
  meta:
    num_workers: -1
  feature_sets:
    eGeMAPSv02: null
    wav2vec2:
      path_suffix: "audeering"  # will be popped
      variant: "wav2vec2-large-robust-12-ft-emotion-msp-dim"
      num_hidden_layers: 0
      device: "cpu"  # will be popped

DataProcessor:
  meta:
    try_load_existing_split: False
  splitting:
    fixed_test_speakers:
      speaker_column: "participant_code"
      test_speakers: {test_speakers}

ModelTrainer:
  meta:
      num_workers: -1
      predict_proba: False
      groups: "participant_code"
      sessions: "session"
      save_full_data: False
  cohorts:
    - "all"
  speech_tasks:
    - "standardized_tasks":
        ["work_tasks_4718", "sustained_utterance_7351", "emotion_acting_5839", "emotion_acting_4373", "count_9822"]
  feature_selections:
    - type: "no_feature_selection"
  feature_normalizations:
    - "sklearn_standard_scaler"
  personalisations:
    - "none":
        null
  target_variables:
    - "phq_8_total_score":
        range_raw: [0, 24]
        range_normalized: [0, 1]
    - "stress_current":
        range_raw: [0, 100]
        range_normalized: [0, 1]
    - "stress_work_tasks":
        range_raw: [0, 100]
        range_normalized: [0, 1]
    - "pss_10_total_score":
        range_raw: [0, 40]
        range_normalized: [0, 1]
    - "who_5_percentage_score_corrected":
        range_raw: [0, 100]
        range_normalized: [0, 1]
  cv_strategies_outer:
    - "loso":
        null
  cv_strategies_inner:
    - "group_k_fold":
        n_splits: 5
    - "k_fold":
        n_splits: 5
        shuffle: True
        random_state: 42
    # - "no_inner_cv":
    #     null
  cv_methods_inner:
    - "GridSearchCV":
       null
    # - "no_inner_cv":
    #     null
  estimators:
    # - type: "LinearRegression"
    #   grid:
    #     fit_intercept: True
    #   problem_type: "regression"
    #   grid_description: "fit_intercept_true"
    - type: "RandomForestRegressor"
      grid:
        # Roughly 100 combinations
        n_estimators: [100, 200]
        max_depth: [10, 20]
        max_features: [1, 'sqrt']
        min_samples_split: [5]
        min_samples_leaf: [1, 2]
        bootstrap: [True]
        # n_estimators: [100, 200]
        # max_depth: [10, 20, null]
        # max_features: [1, 'sqrt']
        # min_samples_split: [2, 5]
        # min_samples_leaf: [1, 2]
        # bootstrap: [True, False]
      problem_type: "regression"
      grid_description: "n_estimators_max_depth_max_features_min_samples_split_min_samples_leaf_bootstrap-small"
'''

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w") as f:
        f.write(config_content)

    return config_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for the active speech pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for synthetic data output. "
            "Defaults to synthetic_data/active/ relative to this script."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=25,
        help="Number of synthetic participants to generate.",
    )
    parser.add_argument(
        "--audio-sr",
        type=int,
        default=16_000,
        help="Sampling rate for synthetic audio (default: 16000).",
    )
    args = parser.parse_args()

    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        output_dir = script_dir
    else:
        output_dir = os.path.abspath(args.output_dir)

    audio_dir = os.path.join(output_dir, "audio")
    csv_path = os.path.join(output_dir, "synthetic_metadata.csv")
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(script_dir)),
        "src",
        "experiment_configs",
        "synthetic",
    )

    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Step 1: Generate metadata
    # -----------------------------------------------------------------------
    print(f"Generating synthetic metadata for {args.n_participants} participants...")
    df_meta, audio_specs = generate_synthetic_metadata(
        n_participants=args.n_participants,
        sessions_range=(3, 12),
        audio_dir=audio_dir,
        rng=rng,
    )

    # Save metadata CSV
    os.makedirs(output_dir, exist_ok=True)
    df_meta.to_csv(csv_path, index=False)
    print(f"  Metadata CSV: {csv_path}")
    print(f"  Shape: {df_meta.shape}")
    print(f"  Participants: {df_meta['participant_code'].nunique()}")
    print(f"  Sessions: {df_meta['session'].nunique()}")
    print(f"  Files: {len(df_meta)}")

    # -----------------------------------------------------------------------
    # Step 2: Generate audio files
    # -----------------------------------------------------------------------
    if sf is not None:
        print(f"\nGenerating {len(audio_specs)} synthetic audio files...")
        for i, (rel_path, duration_s) in enumerate(audio_specs):
            wav_path = os.path.join(audio_dir, rel_path)
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)

            waveform = generate_synthetic_audio(
                duration_s, sr=args.audio_sr, rng=rng
            )
            sf.write(wav_path, waveform, args.audio_sr)

            if (i + 1) % 100 == 0 or (i + 1) == len(audio_specs):
                print(f"  Generated {i + 1}/{len(audio_specs)} files")
    else:
        print(
            "\nSkipping audio generation (soundfile not installed)."
            "\nInstall with: pip install soundfile"
        )

    # -----------------------------------------------------------------------
    # Step 3: Generate experiment config
    # -----------------------------------------------------------------------
    # Pick 3 random participants as fixed test speakers
    unique_participants = df_meta["participant_code"].unique().tolist()
    n_test = min(3, len(unique_participants))
    test_speakers = rng.choice(
        unique_participants, size=n_test, replace=False
    ).tolist()

    config_path = generate_experiment_config(
        output_dir=config_dir,
        csv_path=csv_path,
        audio_dir=audio_dir,
        test_speakers=test_speakers,
    )
    print(f"\nExperiment config: {config_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory:   {output_dir}")
    print(f"  Metadata CSV:       {csv_path}")
    print(f"  Audio directory:    {audio_dir}")
    print(f"  Experiment config:  {config_path}")
    print(f"  Participants:       {df_meta['participant_code'].nunique()}")
    print(f"  Sessions:           {df_meta['session'].nunique()}")
    print(f"  Audio files:        {len(audio_specs)}")
    print(f"  Test speakers:      {test_speakers}")
    print()
    print("To run the pipeline with synthetic data:")
    print(f"  python src/main.py {os.path.relpath(config_path, os.path.dirname(os.path.dirname(script_dir)))}")
    print()
    print(
        "NOTE: Results from synthetic data have NO scientific meaning.\n"
        "      This is for pipeline verification only."
    )


if __name__ == "__main__":
    main()
