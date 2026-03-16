#!/usr/bin/env python3
"""
Generate synthetic data for the passive VDR audio analysis pipeline.

Creates:
  1. Synthetic FLAC audio files organised by daily directories and microphone channels
  2. Synthetic wind speed CSV
  3. Synthetic event timings YAML (land/sea periods, loading/discharge)
  4. Synthetic pre-computed prediction pickle files for the evaluation pipeline
  5. Synthetic active survey data (shared with active pipeline) for mediation analysis

Privacy notice:
  - All timestamps use a completely synthetic date range (2024-01-15 to 2024-02-28)
  - All participant IDs are randomly generated UUIDs
  - Audio files are purely synthetic (white noise + sine waves, NOT speech)
  - Prediction values are random (not matching real distributions)
  - No real data can be reconstructed from these synthetic samples

Usage:
    python generate_synthetic_data.py [--output-dir OUTPUT_DIR] [--seed SEED]
                                       [--n-days N_DAYS]
"""

import argparse
import os
import warnings
from datetime import datetime, timedelta

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

try:
    import yaml
except ImportError:
    yaml = None
    warnings.warn(
        "PyYAML not installed. YAML files will not be generated. "
        "Install with: pip install pyyaml"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MICROPHONES = ["M1", "M2", "M3", "M6", "V4", "V5"]
BRIDGE_MICS = ["M1", "M2", "M3", "M6"]
VESSEL_ID = "0000000_0"  # Synthetic vessel ID
AUDIO_SR = 16_000  # Sampling rate for passive audio (16 kHz)
EMOTION_DIMS = ["prediction_arousal", "prediction_dominance", "prediction_valence"]


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

    NOT speech - purely for pipeline structural verification.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    noise = rng.normal(0, 0.05, n_samples).astype(np.float32)
    n_tones = rng.integers(1, 4)
    signal = noise.copy()
    for _ in range(n_tones):
        freq = rng.uniform(50, 3000)
        amplitude = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        signal += (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(
            np.float32
        )

    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * 0.9

    return signal


# ---------------------------------------------------------------------------
# FLAC audio file generation (VDR structure)
# ---------------------------------------------------------------------------
def generate_vdr_audio_files(output_dir, days, rng, sr=16_000, segments_per_day=8):
    """
    Generate synthetic FLAC files mimicking VDR directory structure.

    Directory structure: data/interim/IMO_{vessel}_{timestamp}/YYMMDD,HHMMSS,{mic},{vessel}.flac

    Parameters
    ----------
    output_dir : str
        Root output directory.
    days : list of datetime
        List of days to generate data for.
    rng : numpy.random.Generator
        Random number generator.
    sr : int
        Sampling rate.
    segments_per_day : int
        Number of audio segments per microphone per day.

    Returns
    -------
    list
        List of generated file paths (relative to output_dir).
    """
    if sf is None:
        print("  Skipping FLAC generation (soundfile not installed)")
        return []

    interim_dir = os.path.join(output_dir, "data", "interim")
    generated_files = []

    for day in days:
        day_str = day.strftime("%y%m%d")
        # Create a daily directory
        dir_name = f"IMO_{VESSEL_ID}_{day.strftime('%Y%m%d')}"
        day_dir = os.path.join(interim_dir, dir_name)
        os.makedirs(day_dir, exist_ok=True)

        for seg_idx in range(segments_per_day):
            # Random time within the day
            hour = rng.integers(0, 24)
            minute = rng.integers(0, 60)
            second = rng.integers(0, 60)
            time_str = f"{hour:02d}{minute:02d}{second:02d}"

            for mic in MICROPHONES:
                filename = f"{day_str},{time_str},{mic},{VESSEL_ID}.flac"
                filepath = os.path.join(day_dir, filename)

                # Generate 5-15 seconds of synthetic audio
                duration = rng.uniform(5.0, 15.0)
                waveform = generate_synthetic_audio(duration, sr=sr, rng=rng)
                sf.write(filepath, waveform, sr)
                generated_files.append(
                    os.path.relpath(filepath, output_dir)
                )

    return generated_files


# ---------------------------------------------------------------------------
# Wind speed data
# ---------------------------------------------------------------------------
def generate_wind_speed_csv(output_dir, days, rng):
    """
    Generate synthetic wind speed data.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    days : list of datetime
        List of days.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    wind_dir = os.path.join(output_dir, "data", "evaluated", "wind_speed")
    os.makedirs(wind_dir, exist_ok=True)

    rows = []
    for day in days:
        # Generate measurements every 10 minutes
        for hour in range(24):
            for minute in range(0, 60, 10):
                ts = day.replace(hour=hour, minute=minute, second=0)
                # Synthetic wind speed: base level + daily variation + noise
                base_wind = rng.uniform(5, 25)
                daily_component = 5 * np.sin(2 * np.pi * hour / 24)
                noise = rng.normal(0, 2)
                wind_speed = max(0, base_wind + daily_component + noise)
                rows.append({"time": ts.isoformat(), "true_wind_speed_gps": wind_speed})

    df_wind = pd.DataFrame(rows)
    csv_path = os.path.join(wind_dir, "true_wind_speed.csv")
    df_wind.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Event timings YAML
# ---------------------------------------------------------------------------
def generate_event_timings(output_dir, days, rng):
    """
    Generate synthetic event timings YAML.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    days : list of datetime
        List of days.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    str
        Path to the generated YAML file.
    """
    if yaml is None:
        print("  Skipping event timings YAML (PyYAML not installed)")
        return None

    n_days = len(days)
    # Divide the period into: port1 -> sea1 -> port2 (loading) -> sea2 -> port3 (discharge)
    splits = sorted(rng.choice(range(2, n_days - 2), size=4, replace=False))

    events = {
        "land_and_sea": {
            "land": [
                {
                    "start": days[0].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": days[splits[0]].strftime("%Y-%m-%d %H:%M:%S"),
                    "location": "Port_A",
                },
                {
                    "start": days[splits[1]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": days[splits[2]].strftime("%Y-%m-%d %H:%M:%S"),
                    "location": "Port_B",
                },
                {
                    "start": days[splits[3]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": days[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    "location": "Port_C",
                },
            ],
            "sea": [
                {
                    "start": days[splits[0]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": days[splits[1]].strftime("%Y-%m-%d %H:%M:%S"),
                    "location": "sea",
                },
                {
                    "start": days[splits[2]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": days[splits[3]].strftime("%Y-%m-%d %H:%M:%S"),
                    "location": "sea",
                },
            ],
        },
        "loading_and_discharge": {
            "loading": [
                {
                    "start": days[splits[1]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": (days[splits[1]] + timedelta(hours=18)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "location": "Port_B",
                },
            ],
            "discharge": [
                {
                    "start": days[splits[3]].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": (days[splits[3]] + timedelta(hours=12)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "location": "Port_C",
                },
            ],
        },
    }

    yaml_dir = os.path.join(output_dir, "src", "evaluate_results")
    os.makedirs(yaml_dir, exist_ok=True)
    yaml_path = os.path.join(yaml_dir, "evaluate_time_course_events.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(events, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# ---------------------------------------------------------------------------
# Prediction pickle files (pre-computed intermediate results)
# ---------------------------------------------------------------------------
def generate_prediction_pickles(output_dir, days, rng):
    """
    Generate synthetic prediction DataFrames mimicking the pipeline output.

    These pickle files are the intermediate outputs that the evaluation
    scripts consume (emotion predictions per segment, SNR predictions).

    Parameters
    ----------
    output_dir : str
        Root output directory.
    days : list of datetime
        List of days.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Paths to the generated pickle files.
    """
    evaluated_dir = os.path.join(output_dir, "data", "evaluated")
    os.makedirs(evaluated_dir, exist_ok=True)

    # Also create per-directory prediction pickles in output/
    output_pred_dir = os.path.join(output_dir, "data", "output")

    all_emotion_rows = []
    all_snr_rows = []

    for day in days:
        day_str = day.strftime("%y%m%d")
        dir_name = f"IMO_{VESSEL_ID}_{day.strftime('%Y%m%d')}"

        # Create per-directory predictions
        pred_dir_emotion = os.path.join(
            output_pred_dir, dir_name, "predictions-emotion"
        )
        pred_dir_snr = os.path.join(
            output_pred_dir, dir_name, "predictions-snr"
        )
        os.makedirs(pred_dir_emotion, exist_ok=True)
        os.makedirs(pred_dir_snr, exist_ok=True)

        dct_emotion = {}
        dct_snr = {}

        n_segments = rng.integers(20, 60)

        for mic in MICROPHONES:
            segments = []
            snr_segments = []

            for _ in range(n_segments):
                hour = rng.integers(0, 24)
                minute = rng.integers(0, 60)
                second = rng.integers(0, 60)
                time_str = f"{hour:02d}{minute:02d}{second:02d}"

                filename = f"{day_str},{time_str},{mic},{VESSEL_ID}.flac"
                start = pd.Timedelta(seconds=float(rng.uniform(0, 2)))
                duration = rng.uniform(1.0, 6.0)
                end = start + pd.Timedelta(seconds=duration)

                # Emotion predictions (arousal, dominance, valence in [0, 1])
                segments.append(
                    {
                        "file": filename,
                        "start": start,
                        "end": end,
                        "prediction_arousal": float(rng.uniform(0.3, 0.7)),
                        "prediction_dominance": float(rng.uniform(0.3, 0.7)),
                        "prediction_valence": float(rng.uniform(0.3, 0.7)),
                    }
                )

                # SNR predictions
                snr_segments.append(
                    {
                        "file": filename,
                        "start": start,
                        "end": end,
                        "snr": float(rng.uniform(-5, 30)),
                    }
                )

            df_mic_emotion = pd.DataFrame(segments)
            if len(df_mic_emotion) > 0:
                df_mic_emotion = df_mic_emotion.set_index(["file", "start", "end"])
            dct_emotion[mic] = df_mic_emotion

            df_mic_snr = pd.DataFrame(snr_segments)
            if len(df_mic_snr) > 0:
                df_mic_snr = df_mic_snr.set_index(["file", "start", "end"])
            dct_snr[mic] = df_mic_snr

            # Accumulate for the concatenated DataFrame
            for seg in segments:
                seg_copy = dict(seg)
                seg_copy["microphone"] = mic
                all_emotion_rows.append(seg_copy)
            for seg in snr_segments:
                seg_copy = dict(seg)
                seg_copy["microphone"] = mic
                all_snr_rows.append(seg_copy)

        # Save per-directory pickle files
        emotion_pkl_name = f"dct_predictions_{dir_name}.pkl"
        snr_pkl_name = f"dct_predictions_{dir_name}.pkl"
        pd.to_pickle(dct_emotion, os.path.join(pred_dir_emotion, emotion_pkl_name))
        pd.to_pickle(dct_snr, os.path.join(pred_dir_snr, snr_pkl_name))

    # Build concatenated DataFrames (what evaluate_time_course_utils.concat_all_preds produces)
    df_all_emotion = pd.DataFrame(all_emotion_rows)
    df_all_emotion = df_all_emotion.set_index(["file", "start", "end"])

    # Extract datetime from filename and add time/microphone columns
    def extract_datetime_from_filename(filename):
        """Extract datetime from VDR filename like '230205,015956,M1,9510682_0.flac'."""
        parts = filename.split(",")
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1]
            try:
                return pd.to_datetime(date_str + time_str, format="%y%m%d%H%M%S")
            except Exception:
                return pd.NaT
        return pd.NaT

    filenames = df_all_emotion.index.get_level_values("file")
    df_all_emotion["time"] = [extract_datetime_from_filename(f) for f in filenames]
    # noise_status: None means original (non-denoised) audio -- required by
    # evaluate_time_course_main.py which groups by this column.
    df_all_emotion["noise_status"] = None

    df_all_snr = pd.DataFrame(all_snr_rows)
    df_all_snr = df_all_snr.set_index(["file", "start", "end"])
    filenames_snr = df_all_snr.index.get_level_values("file")
    df_all_snr["time"] = [extract_datetime_from_filename(f) for f in filenames_snr]
    df_all_snr["noise_status"] = None

    # Save concatenated pickles
    emotion_concat_path = os.path.join(evaluated_dir, "df_all_predictions_emotion.pkl")
    snr_concat_path = os.path.join(evaluated_dir, "df_all_predictions_snr.pkl")

    df_all_emotion.to_pickle(emotion_concat_path)
    df_all_snr.to_pickle(snr_concat_path)

    # Also save a non-denoised version (used by confounder analysis)
    noden_path = os.path.join(
        evaluated_dir,
        "df_all_predictions_emotion-no_denoising-auvad-mobilenet-no_transcripts.pkl",
    )
    # Slightly perturb the emotion values
    df_noden = df_all_emotion.copy()
    for col in EMOTION_DIMS:
        df_noden[col] = df_noden[col] + rng.normal(0, 0.02, len(df_noden))
        df_noden[col] = df_noden[col].clip(0, 1)
    df_noden.to_pickle(noden_path)

    print(f"  Emotion predictions: {len(df_all_emotion)} segments")
    print(f"  SNR predictions:     {len(df_all_snr)} segments")

    return {
        "emotion_concat": emotion_concat_path,
        "snr_concat": snr_concat_path,
        "noden_concat": noden_path,
    }


# ---------------------------------------------------------------------------
# Active survey data for mediation analysis
# ---------------------------------------------------------------------------
def generate_active_survey_data(output_dir, days, rng, n_participants=15,
                                active_csv_path=None):
    """
    Provide active survey data for the mediation analysis.

    If ``active_csv_path`` points to an existing file (e.g. the CSV produced
    by the active pipeline generator), that file is reused so that both
    pipelines share the same participant identifiers.  Otherwise, standalone
    synthetic survey data is generated.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    days : list of datetime
        List of days.
    rng : numpy.random.Generator
        Random number generator.
    n_participants : int
        Number of synthetic participants (used only when generating fresh data).
    active_csv_path : str or None
        Path to an existing active metadata CSV to reuse.

    Returns
    -------
    str
        Path to the generated/copied CSV file.
    """
    csv_path = os.path.join(output_dir, "data", "evaluated", "synthetic_active_data.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Reuse the active pipeline CSV when available
    if active_csv_path and os.path.isfile(active_csv_path):
        import shutil
        shutil.copy2(active_csv_path, csv_path)
        print(f"  Reused active metadata from {active_csv_path}")
        return csv_path

    participant_codes = [_seeded_uuid(rng) for _ in range(n_participants)]
    speaker_files = [_seeded_uuid(rng) for _ in range(n_participants)]

    rows = []
    for p_idx in range(n_participants):
        # Each participant contributes data on a subset of days
        n_sessions = rng.integers(3, min(len(days), 15))
        session_days = sorted(rng.choice(days, size=n_sessions, replace=False))

        for s_idx, day in enumerate(session_days):
            session_id = f"session_{p_idx:03d}_{s_idx:03d}"
            # Generate multiple files per session (5 prompts)
            for prompt_idx in range(5):
                file_uuid = _seeded_uuid(rng)
                rel_path = os.path.join(
                    speaker_files[p_idx], session_id, f"{file_uuid}.wav"
                )
                rows.append(
                    {
                        "file": rel_path,
                        "participant_code": participant_codes[p_idx],
                        "session": session_id,
                        "speaker_file": speaker_files[p_idx],
                        "survey": rng.choice(
                            ["daily", "weekly_01", "weekly_02"]
                        ),
                        "prompt": rng.choice(
                            [
                                "work_tasks_4718",
                                "sustained_utterance_7351",
                                "emotion_acting_5839",
                                "emotion_acting_4373",
                                "count_9822",
                            ]
                        ),
                        "date_file": day.strftime("%Y-%m-%d"),
                        "date_survey": day.strftime("%Y-%m-%d"),
                        "age": int(rng.integers(22, 62)),
                        "sex": rng.choice(["male", "female"]),
                        "stress_current": float(rng.uniform(0, 100)),
                        "stress_work_tasks": float(rng.uniform(0, 100)),
                        "pss_10_total_score": float(rng.integers(0, 41)),
                        "phq_8_total_score": float(rng.integers(0, 25)),
                        "who_5_percentage_score_corrected": float(
                            rng.uniform(0, 100)
                        ),
                        "emotions_current_happy": float(rng.uniform(0, 100)),
                        "emotions_current_sad": float(rng.uniform(0, 100)),
                        "emotions_current_anger": float(rng.uniform(0, 100)),
                        "emotions_current_fear": float(rng.uniform(0, 100)),
                        "emotions_current_indifference": float(
                            rng.uniform(0, 100)
                        ),
                        "emotions_current_shame": float(rng.uniform(0, 100)),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for the passive VDR audio pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Root directory for synthetic data output. "
            "Defaults to synthetic_data/passive/ relative to this script."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=40,
        help="Number of synthetic days to generate (default: 40, matching the real voyage).",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip generating FLAC audio files (only generate metadata and pickles).",
    )
    args = parser.parse_args()

    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        output_dir = script_dir
    else:
        output_dir = os.path.abspath(args.output_dir)

    rng = np.random.default_rng(args.seed)

    # Generate date range
    base_date = datetime(2024, 1, 15)
    days = [base_date + timedelta(days=i) for i in range(args.n_days)]

    print(f"Generating synthetic passive data for {args.n_days} days...")
    print(f"Date range: {days[0].strftime('%Y-%m-%d')} to {days[-1].strftime('%Y-%m-%d')}")
    print(f"Output directory: {output_dir}")

    # -----------------------------------------------------------------------
    # Step 1: Generate FLAC audio files
    # -----------------------------------------------------------------------
    if not args.skip_audio:
        print(f"\nStep 1: Generating VDR FLAC audio files...")
        audio_files = generate_vdr_audio_files(
            output_dir, days, rng, sr=AUDIO_SR, segments_per_day=6
        )
        print(f"  Generated {len(audio_files)} FLAC files")
    else:
        print(f"\nStep 1: Skipping audio generation (--skip-audio)")

    # -----------------------------------------------------------------------
    # Step 2: Generate wind speed CSV
    # -----------------------------------------------------------------------
    print(f"\nStep 2: Generating wind speed data...")
    wind_csv = generate_wind_speed_csv(output_dir, days, rng)
    print(f"  Wind speed CSV: {wind_csv}")

    # -----------------------------------------------------------------------
    # Step 3: Generate event timings YAML
    # -----------------------------------------------------------------------
    print(f"\nStep 3: Generating event timings YAML...")
    events_yaml = generate_event_timings(output_dir, days, rng)
    if events_yaml:
        print(f"  Events YAML: {events_yaml}")

    # -----------------------------------------------------------------------
    # Step 4: Generate prediction pickles
    # -----------------------------------------------------------------------
    print(f"\nStep 4: Generating prediction pickle files...")
    pickle_paths = generate_prediction_pickles(output_dir, days, rng)
    for key, path in pickle_paths.items():
        print(f"  {key}: {path}")

    # -----------------------------------------------------------------------
    # Step 5: Generate active survey data
    # -----------------------------------------------------------------------
    print(f"\nStep 5: Providing active survey data for mediation analysis...")
    # Reuse the active pipeline CSV if it exists (avoids duplicate participants)
    active_meta_candidate = os.path.join(
        os.path.dirname(script_dir), "active", "synthetic_metadata.csv"
    )
    active_csv = generate_active_survey_data(
        output_dir, days, rng, active_csv_path=active_meta_candidate
    )
    print(f"  Active CSV: {active_csv}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SYNTHETIC PASSIVE DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory:    {output_dir}")
    print(f"  Days generated:      {args.n_days}")
    print(f"  Microphone channels: {', '.join(MICROPHONES)}")
    if events_yaml:
        print(f"  Events YAML:         {events_yaml}")
    print(f"  Wind speed CSV:      {wind_csv}")
    print(f"  Active survey CSV:   {active_csv}")
    print()
    print(
        "NOTE: Results from synthetic data have NO scientific meaning.\n"
        "      This is for pipeline verification only."
    )


if __name__ == "__main__":
    main()
