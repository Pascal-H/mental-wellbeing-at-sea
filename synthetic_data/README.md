# Synthetic Data for Pipeline Verification

This directory contains scripts to generate **synthetic data** that allows
readers to run both the active speech modelling pipeline and the passive VDR
audio analysis pipeline without access to the real (private) dataset.

> **Privacy notice:** All generated data is entirely synthetic.
> Participant IDs are random UUIDs, labels are drawn from uniform distributions
> (not matching real data distributions), and audio files contain white noise
> plus sine tones (no speech). No real data can be reconstructed from these
> synthetic samples.

## Quick Start

### Prerequisites

```bash
pip install numpy pandas soundfile pyyaml
```

### Generate Active Pipeline Data

```bash
# From the repository root (modelling-static/):
python synthetic_data/active/generate_synthetic_data.py

# Or with custom parameters:
python synthetic_data/active/generate_synthetic_data.py \
    --n-participants 20 \
    --seed 42
```

This creates:
- `synthetic_data/active/synthetic_metadata.csv`  -  metadata CSV with participant
  demographics, session info, and target variables
- `synthetic_data/active/audio/`  -  directory tree of synthetic WAV files (16 kHz mono)
- `src/experiment_configs/synthetic/synthetic-eGeMAPSv02.yaml`  -  experiment config
  pointing to the generated synthetic data

**Run the active pipeline with synthetic data:**

```bash
python src/main.py src/experiment_configs/synthetic/synthetic-eGeMAPSv02.yaml
```

### Generate Passive Pipeline Data

```bash
# From the repository root (or passive_recordings/):
python synthetic_data/passive/generate_synthetic_data.py

# Skip audio file generation (faster, generates only metadata + pickles):
python synthetic_data/passive/generate_synthetic_data.py --skip-audio

# With custom parameters (default: 40 days, matching the real voyage):
python synthetic_data/passive/generate_synthetic_data.py \
    --n-days 40 \
    --seed 42
```

This creates:
- `synthetic_data/passive/data/interim/`  -  VDR FLAC audio files organised by
  daily directories and 6 microphone channels (M1, M2, M3, M6, V4, V5)
- `synthetic_data/passive/data/evaluated/wind_speed/true_wind_speed.csv`  -  wind speed data
- `synthetic_data/passive/data/evaluated/df_all_predictions_emotion.pkl`  -  pre-computed
  emotion predictions (concatenated DataFrame)
- `synthetic_data/passive/data/evaluated/df_all_predictions_snr.pkl`  -  pre-computed
  SNR predictions
- `synthetic_data/passive/src/evaluate_results/evaluate_time_course_events.yaml`  - 
  event timings (land/sea periods, loading/discharge)
- `synthetic_data/passive/data/evaluated/synthetic_active_data.csv`  -  active survey
  data for mediation analysis

## What Is Generated

### Active Pipeline

| Component | Description | Format |
|---|---|---|
| Metadata CSV | 15 participants x 3-12 sessions x 5 prompts | CSV (23 columns) |
| Audio files | 2-18 s synthetic signals (noise + sine tones) | WAV, 16 kHz mono |
| Experiment config | YAML pointing to synthetic data paths | YAML |

**Key metadata columns:** `file`, `participant_code`, `session`, `survey`,
`prompt`, `age`, `sex`, `speaker_file`, `date_file`, `date_survey`,
`phq_8_total_score`, `stress_current`, `stress_work_tasks`,
`pss_10_total_score`, `who_5_percentage_score_corrected`,
`pss_10_category`, `who_5_category`, emotion VAS scales.

### Passive Pipeline

| Component | Description | Format |
|---|---|---|
| VDR audio | 40 days x 6 mics x 6 segments/day | FLAC, 16 kHz mono |
| Wind speed | 10-minute interval measurements | CSV |
| Event timings | Port/sea/loading/discharge periods | YAML |
| Emotion predictions | Per-segment arousal/dominance/valence | Pickle (DataFrame) |
| SNR predictions | Per-segment signal-to-noise ratio | Pickle (DataFrame) |
| Active survey data | For mediation analysis merging | CSV |

## Privacy Safeguards

- **No real participant data** is included or derivable
- Participant IDs are freshly generated UUIDs (not hashes of real IDs)
- Target variable values are sampled from **uniform distributions** (not
  matching real data statistics)
- Audio files contain only synthetic waveforms (white noise + sine tones)
- Date ranges are deliberately shifted to a synthetic period (January-February 2024)
- The random seed ensures full reproducibility: participant IDs, file paths,
  and all numeric values are deterministic for a given seed
- If the active pipeline CSV (`synthetic_data/active/synthetic_metadata.csv`)
  exists when the passive generator runs, it is reused as active survey data
  so that both pipelines share the same participant identifiers

## Directory Structure

```
synthetic_data/
|-- README.md                           <- this file
|-- active/
|   |-- generate_synthetic_data.py      <- active pipeline data generator
|   |-- synthetic_metadata.csv          <- (generated) metadata CSV
|   \-- audio/                          <- (generated) WAV files
|       \-- {speaker_uuid}/{session_id}/{file_uuid}.wav
\-- passive/
    |-- generate_synthetic_data.py      <- passive pipeline data generator
    |-- data/
    |   |-- interim/                    <- (generated) VDR FLAC files
    |   |   \-- IMO_{vessel}_{date}/YYMMDD,HHMMSS,{mic},{vessel}.flac
    |   |-- output/                     <- (generated) per-directory predictions
    |   |   \-- IMO_{vessel}_{date}/predictions-{emotion,snr}/
    |   \-- evaluated/
    |       |-- wind_speed/true_wind_speed.csv
    |       |-- df_all_predictions_emotion.pkl
    |       |-- df_all_predictions_snr.pkl
    |       \-- synthetic_active_data.csv
    \-- src/
        \-- evaluate_results/
            \-- evaluate_time_course_events.yaml
```

## Notes

- **Results from synthetic data have no scientific meaning.** The synthetic
  labels are random and unrelated to the audio content. This is strictly for
  verifying that the code runs correctly.
- The active pipeline generator also creates a synthetic experiment configuration
  at `src/experiment_configs/synthetic/synthetic-eGeMAPSv02.yaml`.
- For the passive pipeline evaluation scripts, you may need to adjust the
  hardcoded paths in `passive_recordings/src/evaluate_results/` to point to the
  synthetic data directory instead of the default compute cluster paths.


## Complete Pipeline Verification

The following sequence runs **all** scripts end-to-end with synthetic data.
It was verified to complete successfully. All commands are run from the
repository root (`modelling-static/`).

```bash
# --- Generate synthetic data ---
python synthetic_data/active/generate_synthetic_data.py --n-participants 15 --seed 42
python synthetic_data/passive/generate_synthetic_data.py --n-days 40 --seed 42 --skip-audio

# --- Active speech modelling pipeline ---
python src/main.py src/experiment_configs/synthetic/synthetic-eGeMAPSv02.yaml

# --- Collect results ---
python src/collect_results.py
# -> results/synthetic/composed/everything/

# --- Bootstrap confidence intervals and collect best-performing models ---
# Run the notebooks at
notebooks/mwas/synthetic-bootstrapping-bulk_apply_confidence_intervals_to_results.ipynb
# -> results/synthetic/composed/everything/*-conf.csv
notebooks/mwas/synthetic-paper-collect_results_for_expanded_main_table.ipynb
# -> results/synthetic/composed/compiled-synthetic-paper.csv

# --- Session-level evaluation ---
python notebooks/mwas/evaluate_session_level_ccc.py
python notebooks/mwas/validate_retrospective_labels.py
# Note: evaluate_session_level_ccc-scatterplot.py targets the specific best-performing
# model configuration from the paper (WHO-5/eGeMAPS/SVR) and is not run here.
# -> results/synthetic/composed/session_level_analysis/

# --- Compose LaTeX table ---
notebooks/mwas/synthetic-paper-compose_main_modelling_table-session_level.ipynb

# --- Passive pipeline evaluation ---
python passive_recordings/src/evaluate_results/evaluate_time_course_main.py
python passive_recordings/src/evaluate_results/confounder_noise_denoising_main.py
python passive_recordings/src/evaluate_results/mediation_analysis_main.py --quick
python passive_recordings/src/evaluate_results/mediation_diagram.py \
    --input synthetic_data/passive/data/evaluated/synthetic-mediation-wind_emotion_stress/mediation_results.yaml \
    --output synthetic_data/passive/data/evaluated/synthetic-mediation-wind_emotion_stress/causal_diagram_pgf.pdf
```

### Outputs produced

| Script | Output location |
|---|---|
| Active modelling | `results/synthetic/modelling/` (5 target subdirectories) |
| `collect_results.py` | `results/synthetic/composed/everything/*.csv` |
| `synthetic-bootstrapping-....ipynb` | `results/synthetic/composed/everything/*-conf.csv` |
| `synthetic-paper-collect_....ipynb` | `results/synthetic/composed/compiled-synthetic-paper.csv` |
| Session-level scripts | `results/synthetic/composed/session_level_analysis/` |
| `synthetic-paper-compose_....ipynb` | LaTeX table printed in notebook output |
| Time-course evaluation | `synthetic_data/passive/data/evaluated/synthetic-time_course/` |
| Confounder analysis | `synthetic_data/passive/data/evaluated/synthetic-confounder-noise_denoising/` |
| Mediation analysis | `synthetic_data/passive/data/evaluated/synthetic-mediation-wind_emotion_stress/` |

### Scripts not covered

The upstream audio compression (`passive_recordings/src/compress_extracted_files/`)
and batch prediction (`passive_recordings/src/process_and_predict/`) scripts
require access to real VDR audio files and additional system dependencies
(`sox`) and are therefore not runnable with synthetic data alone.
