# Passive Voyage Data Recorder Analysis

This repository contains the source code for analyzing audio data from voyage data recorders (VDR) to predict perceived emotion expressions in maritime environments. The code processes continuous audio recordings from multiple microphone channels and applies machine learning models for emotion prediction.

## Overview

The analysis pipeline processes audio data from VDR systems to:
1. Convert and compress audio files
2. Apply denoising and voice activity detection (VAD)
3. Predict perceived emotion expressions
4. Generate evaluation plots and correlation analyses

## Data Structure

Due to privacy constraints, the original VDR data cannot be shared. The expected data structure is:
```
data/interim/IMO_[vessel_id]_[timestamp]/[date],[time],[channel],[vessel_id].wav
```

Example: `IMO_9510682_2022-12-25T23-56-48/221225,235648,M1,9510682.wav`

## Processing Pipeline

### 1. Audio Compression ([`src/compress_extracted_files/`](src/compress_extracted_files/))

Converts WAV files from VDR to FLAC format for efficient storage and processing.

- `compress_launcher.py`: Orchestrates the compression process
- `compress_main.py`: Main compression logic
- `compress_utils.py`: Utility functions for audio conversion
- `compress_args_external.py`: Testing configuration

### 2. Audio Processing and Prediction ([`src/process_and_predict/`](src/process_and_predict/))

Processes 6 continuous microphone channels through denoising, VAD, and emotion prediction.

- `batch_predict.sh`: Batch processing script for multiple directories
- `predict_main.py`: Main prediction pipeline
- `predict_utils.py`: Core processing functions (denoising, VAD, emotion prediction)
- `predict_args_external.py`: Testing configuration
- `directories.txt`: List of directories to process

### 3. Results Evaluation ([`src/evaluate_results/`](src/evaluate_results/))

Generates plots and analyses for publication, including SNR (signal-to-noise ratio) distributions and time series of emotion predictions for both actively collected data (from separate study) and passively recorded VDR data.

- `evaluate_time_course_main.py`: Main evaluation orchestrator
- `evaluate_time_course_utils.py`: Time series analysis utilities
- `evaluate_utils_snr_distribution.py`: SNR distribution analysis
- `evaluate_time_course_events.yaml`: Event timeline (anonymized)
- `evaluate_time_course_args_external.py`: Evaluation configuration
- `evaluate_wind.ipynb`: Wind speed calculation from VDR logs
- `evaluate_wind_correlation.ipynb`: Correlation analysis between wind and emotion predictions

### 4. Confounder Analysis: Noise and Denoising ([`src/evaluate_results/`](src/evaluate_results/))

Evaluates potential confounders in the passive emotion analysis pipeline: (1) whether wind speed degrades audio quality (SNR), (2) whether denoising systematically shifts emotion predictions, and (3) whether the denoising effect depends on wind speed or SNR.

- `confounder_noise_denoising_main.py`: Main analysis script (produces YAML and verbose text report)

### 5. Mediation Analysis ([`src/evaluate_results/`](src/evaluate_results/))

Implements a multilevel mediation model examining how environmental conditions (wind speed) affect crew stress through passively sensed bridge emotion intensity, following the Hayes (2018) mediation framework.

- `mediation_analysis_main.py`: Main mediation analysis script (produces YAML and verbose text report)
- `mediation_diagram.py`: Generates the publication-quality causal path diagram (PGF/LaTeX)

## Requirements

- Python 3.9.2
- See `requirements.txt` for complete package dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Compress audio files:**
   ```bash
   python src/compress_extracted_files/compress_launcher.py
   ```

2. **Process and predict emotions:**
   ```bash
   bash src/process_and_predict/batch_predict.sh
   ```

3. **Generate evaluation results:**
   ```bash
   python src/evaluate_results/evaluate_time_course_main.py
   ```

4. **Wind analysis (Jupyter notebooks):**
   - Open `src/evaluate_results/evaluate_wind.ipynb` for wind speed calculations
   - Open `src/evaluate_results/evaluate_wind_correlation.ipynb` for correlation analysis

5. **Confounder analysis (noise, denoising, wind):**
   ```bash
   python src/evaluate_results/confounder_noise_denoising_main.py
   ```
   Results are saved to `data/evaluated/confounder-noise_denoising/`:
   - `confounder_results.yaml`: Structured results
   - `confounder_results_verbose.txt`: Human-readable report with conditional interpretation

6. **Mediation analysis (wind -> emotion -> stress):**
   ```bash
   python src/evaluate_results/mediation_analysis_main.py
   ```
   Results are saved to `data/evaluated/mediation-wind_emotion_stress/`:
   - `mediation_results.yaml`: Structured results for downstream plotting
   - `mediation_results_verbose.txt`: Human-readable report with interpretation

7. **Generate mediation path diagram:**
   ```bash
   python src/evaluate_results/mediation_diagram.py
   ```
   Produces `data/evaluated/mediation-wind_emotion_stress/causal_diagram_pgf.pdf` (requires `pdflatex`).

## Configuration

Key parameters can be modified in the respective `*_args_external.py` files for testing purposes. Production configurations are embedded within the main processing scripts.

## Output

The pipeline generates:
- Processed audio files with denoising and VAD segmentation
- Time series plots of actively collected emotion data (stress measures) with event markers
- SNR (signal-to-noise ratio) distribution plots for raw and denoised speech samples from bridge microphones and radio channels
- Time series plots of passively collected emotion predictions (valence, arousal, dominance) from bridge microphones and radio communication channels with daily box plots and event markers
- Correlation analysis tables between wind speed and emotion dimensions across different operational phases (harbor, sea, cargo operations)

## Publication Figures

This repository includes the figures used in the associated scientific publication:

**Figure 4 - Active Data Time Courses:**
- (a) Current stress: [`data/evaluated/plots-time_course-active/Active - stress_current_by_day-polished-be.pdf`](data/evaluated/plots-time_course-active/Active%20-%20stress_current_by_day-polished-be.pdf)
- (b) Stress during work tasks: [`data/evaluated/plots-time_course-active/Active - stress_work_tasks_by_day-polished-be.pdf`](data/evaluated/plots-time_course-active/Active%20-%20stress_work_tasks_by_day-polished-be.pdf)

**Figure 5 - SNR Distribution:**
- (b) Bridge microphones: [`data/evaluated/plots-snr-distribution-min3s-agg3h/M-min3s-agg3h.pdf`](data/evaluated/plots-snr-distribution-min3s-agg3h/M-min3s-agg3h.pdf)
- (c) Radio communication channels: [`data/evaluated/plots-snr-distribution-min3s-agg3h/V-min3s-agg3h.pdf`](data/evaluated/plots-snr-distribution-min3s-agg3h/V-min3s-agg3h.pdf)

**Figure 8 - Bridge Microphones Time Courses:**
- (a) Valence: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_valence_by_day_bridge_M-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_valence_by_day_bridge_M-polished.pdf)
- (b) Arousal: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_arousal_by_day_bridge_M-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_arousal_by_day_bridge_M-polished.pdf)
- (c) Dominance: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_dominance_by_day_bridge_M-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_dominance_by_day_bridge_M-polished.pdf)

**Figure 9 - Radio Communication Time Courses:**
- (a) Valence: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_valence_by_day_radio_V-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_valence_by_day_radio_V-polished.pdf)
- (b) Arousal: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_arousal_by_day_radio_V-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_arousal_by_day_radio_V-polished.pdf)
- (c) Dominance: [`data/evaluated/plots-time_course-passive-emotion/Passive - prediction_dominance_by_day_radio_V-polished.pdf`](data/evaluated/plots-time_course-passive-emotion/Passive%20-%20prediction_dominance_by_day_radio_V-polished.pdf)

**Table 5 - Wind Speed Correlation Analysis:**
- Correlation analysis code: [`src/evaluate_results/evaluate_wind_correlation.ipynb`](src/evaluate_results/evaluate_wind_correlation.ipynb)

**Confounder Analysis - Wind, SNR, and Denoising:**
- Structured results: [`data/evaluated/confounder-noise_denoising/confounder_results.yaml`](data/evaluated/confounder-noise_denoising/confounder_results.yaml)
- Verbose report: [`data/evaluated/confounder-noise_denoising/confounder_results_verbose.txt`](data/evaluated/confounder-noise_denoising/confounder_results_verbose.txt)

**Figure 10 - Mediation Path Diagram:**
- Causal diagram (wind -> emotion -> stress): [`data/evaluated/mediation-wind_emotion_stress/causal_diagram_pgf.pdf`](data/evaluated/mediation-wind_emotion_stress/causal_diagram_pgf.pdf)
- Structured results: [`data/evaluated/mediation-wind_emotion_stress/mediation_results.yaml`](data/evaluated/mediation-wind_emotion_stress/mediation_results.yaml)
- Verbose report: [`data/evaluated/mediation-wind_emotion_stress/mediation_results_verbose.txt`](data/evaluated/mediation-wind_emotion_stress/mediation_results_verbose.txt)

## Citation

Please cite the associated scientific publication when using this code.
