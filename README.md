Mental Wellbeing at Sea: a Prototype to Collect Speech Data in Maritime Settings
==============================

This repository accompanies the publication "Mental Wellbeing at Sea: a Prototype to Collect Speech Data in Maritime Settings" at HEALTHINF 2025:  
[Pascal Hecker](https://orcid.org/0000-0001-6604-1671), [Monica Gonzalez-Machorro](https://orcid.org/0009-0008-9188-058X), [Hesam Sagha](https://orcid.org/0000-0002-8644-9591), [Saumya Dudeja](https://orcid.org/0009-0003-5397-1759), [Matthias Kahlau](https://orcid.org/0009-0004-7017-541X), [Florian Eyben](https://orcid.org/0009-0003-0330-8545), [Bjorn W. Schuller](https://orcid.org/0000-0002-6478-8699), [Bert Arnrich](https://orcid.org/0000-0001-8380-7667)  
In Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOSTEC 2025) - Volume 2: HEALTHINF, pages 29-40  
ISBN: 978-989-758-731-3; ISSN: 2184-4305


Further, it contains the material for the publication under review "Mental Wellbeing at Sea: Active and Passive Speech Monitoring in a Maritime Setting".  
The associated material and its own README can be found under [passive_recordings/](passive_recordings/).


## Publication (HEALTHINF)

The figures presented in the publication are residing in the [figures/](notebooks/mwas/figures) folder, which was mostly composed with the Jupyter notebook [paper-plots_survey_responses.ipynb](notebooks/mwas/paper-plots_survey_responses.ipynb).  
The central table with the significantly correlating features is [paper-significantly_correlating_features.csv](notebooks/mwas/data/feature_analysis/paper-significantly_correlating_features.csv).  
The central table for the statistical modelling approaches is called [compiled-merged_denoised_noisy-paper.ods](results/mwas/composed/compiled-merged_denoised_noisy-paper.ods) and it is being converted to the LaTeX table in the publication with the Jupyter notebook [paper-compose_main_modelling_table.ipynb](notebooks/mwas/paper-compose_main_modelling_table.ipynb).


### Publication (Journal)

For the extended journal publication, the classification performance is provided in session-level aggregates (in the HEALTHINF publication, the results were given for segment-level classification). The notebook [`paper-collect_results_for_expanded_main_table.ipynb`](notebooks/mwas/paper-collect_results_for_expanded_main_table.ipynb) automises the collection of the best-performing models and stores the result as [`compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv`](results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv). Then, the notebook [`paper-compose_main_modelling_table-proper_loso-expanded-session_level.ipynb`](notebooks/mwas/paper-compose_main_modelling_table-proper_loso-expanded-session_level.ipynb) takes that [`compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv`](results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv) file and converts it into the LaTeX source used in the publication. Further, the regression scatter plot of the best-performing model with session-level performance is being generated in [`evaluate_session_level_ccc-scatterplot.py`](notebooks/mwas/evaluate_session_level_ccc-scatterplot.py) and the resulting plot is saved as [`regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.pdf`](notebooks/mwas/figures/regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.pdf).
The additional analyses of the active speech data modelling are described in the section [`Session-level evaluation`](#session-level-evaluation).  
The additional analyses of the passively collected speech data are outlined in the sections [4. Confounder Analysis: Noise and Denoising](passive_recordings/README.md#4-confounder-analysis-noise-and-denoising-srcevaluate_results)  and [5. Mediation Analysis](passive_recordings/README.md#5-mediation-analysis-srcevaluate_results) , as well ass the points `6. Mediation analysis (wind -> emotion -> stress)` and `7. Generate mediation path diagram` in the [`Usage`](passive_recordings/README.md#usage) section in the [`passive_recordings/`](passive_recordings/) subdirectory.


## Getting started

Unfortunately, we cannot share the data due to privacy constraints. The source in this repository was used to run the analyses presented in the publication and should provide some valuable means to check the routines applied.  

To utilise the source code provided in this repository, preferably use a virtual environment manager of your choice and run `pip install -r requirements-freeze-devaice.txt`.  
[devAIce](https://www.audeering.com/products/devaice/) is a commercial framework that provides the voice activity detection (VAD) and the signal-to-noise ratio (SNR) prediction in this study.  
Without a respective devAIce license, you can run `pip install -r requirements-freeze.txt` and have to implement alternative solutions for its functionalities.  
Python version 3.8.10 was used in this project.


## Workflow

### Entry point

The main modelling pipeline launcher script resides in [src/main.py](src/main.py).  
Simply launch it by running `python main.py`.  

In [src/experiment_configs/](src/experiment_configs/), you can find designated configuration files for particular experiment runs.  
A config can be passed through the command line to the main script, such as:  
`python main.py experiment_configs/mental_wellbeing_at_sea/eGeMAPSv02-target_norm-no_denoising.yaml`.  
The section [experiment configs used](#experiment-configs-used) lists all the configuration files that were used to obtain the results presented in the publication.


### Collecting results

The results will be saved in a `results/mwas/modelling` folder and contain a nested folder structure that encodes the respective experiment settings.  
After several experiment runs, you can adjust and execute [src/collect_results.py](src/collect_results.py).  
It will find all `results-compiled.yaml` files and add its evaluation metrics to a .csv file saved in `results/mwas/composed`.  
In the script, you have to manually set the target variable, whose results you want to collect. You can further filter any string that is contained in the results paths and specifies the respective run (e.g., "type-no_feature_selection" for all models for which no feature selection was performed). This is implemented in the `search_term = {"term": None, "name": "everything"}` dictionary, where term would be "type-no_feature_selection" and "name" can be chosen by you to be a recognizable identifier in the `results/mwas/composed` folder hierarchy.

With that .csv file, you can then e.g., open it in LibriOffice, select everything (ctrl + a) &rarr; "Data" &rarr; "Sort" &rarr; "Sorty Key 1" = the column with the metric you find most meaningful, e,g, "Column B" for CCC, select "Descending".  
That was, you get all your models sorted by their performance!  

Then, select your best performing model, or any other model you want to inspect, and scroll to the "path" column. Use that path to navigate, (e.g., using `cd` from `results/mwas/modelling/`) to the model directory and check out the **plots** in the folder for regression plots of the train or test partition prediction.

Alternatively, the notebook [`paper-collect_results_for_expanded_main_table.ipynb`](notebooks/mwas/paper-collect_results_for_expanded_main_table.ipynb) automises the collection of the best-performing models and stores the result as [`compiled-merged_denoised_noisy-paper-proper_loso-expanded`](results/mwas/composed/compiled-merged_denoised_noisy-paper-proper_loso-expanded.csv).


#### Saved predictions and labels

Each model result folder contains a `data` folder. That folder in turn contains `df_results_train.parquet.zstd` and `df_results_test.parquet.zstd`. These parquet files (for good compression) can be read with:  
```
df = pd.read_parquet('df_results_test.parquet.zstd', engine='pyarrow')
```
and `pyarrow` will be installed already through the requirements.
These results DataFrames contain the predicted and ground truth labels, as well as several other useful columns such as the speaker ID and the index of the outer CV fold.

If more in-depth debugging is required, the following option in the experiment configuration saves even further data:  
```

ModelTrainer:
  meta:
      save_full_data: True
```
This will also save the filtered feature- and label DataFrame - to check e.g., how many feature columns were dropped through feature selection or how the feature values were normalized.


#### Confidence intervals

The Jupyter notebook [bootstrapping-bulk_apply_confidence_intervals_to_results.ipynb](notebooks/mwas/bootstrapping-bulk_apply_confidence_intervals_to_results.ipynb) was used to calculate the confidence intervals with the [confidence_intervals](https://github.com/luferrer/ConfidenceIntervals) package.


## Audio-quality-based filtering and denoising

To compare the performance of some denoising methods, a model to estimate the SNR level of the individual audio files was employed in [audio_quality-snr_filtering.ipynb](notebooks/mwas/audio_quality-snr_filtering.ipynb). In the publication, we discard files with an SNR value \< 7. The respective files to filter out were copied to [audio_quality-filter_samples.ipynb](notebooks/mwas/audio_quality-filter_samples.ipynb), and that notebook is processed in the main modelling pipeline in [src/main.py#L317](src/main.py#L317).  

For denoising, the "causal speech enhancement model" ([publication](https://arxiv.org/abs/2006.12847), [repository](https://github.com/facebookresearch/denoiser)) was employed, decoupled from this repository. The resulting file hierarchy was passed back to the pipeline by pointing the configuration file field `path_data` to the directory tree that was denoised; as an example, see the configuration file [eGeMAPSv02-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml#8](src/experiment_configs/mental_wellbeing_at_sea/eGeMAPSv02-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml#8).

The Jupyter notebook [check_clipping.ipynb](notebooks/mwas/check_clipping.ipynb) was employed to check if the denoised files still contain clipping. No denoised files are clipped, but 17 "noisy" files showed clipping.


## Session-level evaluation

Two scripts in `notebooks/mwas/` evaluate model performance at the session level (rather than the segment level used during training):

* [**`evaluate_session_level_ccc.py`**](notebooks/mwas/evaluate_session_level_ccc.py) - Aggregates segment-level predictions to session level by averaging predictions within each (participant, session) pair, then computes CCC with 95% bootstrap confidence intervals for all 20 paper configurations. Validates against the paper's segment-level CCC and the existing session-level CCC (`concordance_cc-test-agg-average`).
  * The regression scatter plot of the best-performing model with session-level performance is being generated in [**`evaluate_session_level_ccc-scatterplot.py`**](notebooks/mwas/evaluate_session_level_ccc-scatterplot.py) and the resulting plot is saved as [`regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.pdf`](notebooks/mwas/figures/regression_who_5_percentage_score_corrected_noisy_eGeMAPSv02-publication.pdf).
* [**`validate_retrospective_labels.py`**](notebooks/mwas/validate_retrospective_labels.py) - Compares session-level CCC on all sessions against assessment-only sessions (where a questionnaire was actually completed) for the 12 retrospective target configurations (WHO-5, PSS-10, PHQ-8). Verifies that retrospective label assignment does not inflate predictive performance.

Both scripts write their output to [`results/mwas/composed/session_level_analysis/`](results/mwas/composed/session_level_analysis/).


## Compile the LaTeX


## Experiment configs used

The experiment configuration files used to run the modelling for the publication are:  

eGeMAPS features
* No denoising: [eGeMAPSv02-target_norm-no_denoising.yaml](src/experiment_configs/mental_wellbeing_at_sea/eGeMAPSv02-target_norm-no_denoising.yaml)
* Denoising and SNR-based filtering: [eGeMAPSv02-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml](src/experiment_configs/mental_wellbeing_at_sea/eGeMAPSv02-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml)

wav2vec2.0 embeddings as features
* No denoising: [wav2vec2-large-robust-12-ft-emotion-msp-target_norm-dim-no_denoising.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-robust-12-ft-emotion-msp-target_norm-dim-no_denoising.yaml)
* Denoising and SNR-based filtering: [wav2vec2-large-robust-12-ft-emotion-msp-dim-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-robust-12-ft-emotion-msp-dim-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml)
* No denoising: [wav2vec2-large-robust-ft-libri-960h-target_norm-no_denoising.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-robust-ft-libri-960h-target_norm-no_denoising.yaml)
* Denoising and SNR-based filtering: [wav2vec2-large-robust-ft-libri-960h-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-robust-ft-libri-960h-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml)
* No denoising: [wav2vec2-large-xlsr-53-target_norm-no_denoising.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-xlsr-53-target_norm-no_denoising.yaml)
* Denoising and SNR-based filtering: [wav2vec2-large-xlsr-53-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml](src/experiment_configs/mental_wellbeing_at_sea/wav2vec2-large-xlsr-53-target_norm-facebook_denoiser-master64-converted_int16_dithering-filter_clipping_snr.yaml)


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    │
    ├── notebooks          <- Jupyter notebooks essential to this repository.
    │   │
    │   ├── data           <- Data assisting the notebooks.
    │   │
    │   └── figures        <- Generated graphics and figures used in the publication.
    │
    ├── requirements-freeze.txt <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to download or generate data.
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │
        └── models         <- Scripts to train models and then use trained models to make
                              predictions.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
