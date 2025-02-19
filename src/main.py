import argparse
import itertools
import os
import pandas as pd
from pathlib import Path
import warnings
import yaml

import audb

from utils import compose_experiment_str
from data.audio_transforms import (
    config_to_transforms,
    apply_transforms,
    apply_vad,
)
from features.feature_extractor import extract_features
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer


def main(path_config):
    """
    Entry point of the program.
    Loads experiment configuration, processes audio data, performs feature extraction,
    splits the data into train/dev/test sets, trains models, and saves the results.
    """
    # Get the directory of the current script
    cwd = Path(__file__).parent

    # Load the experiment configuration from the YAML file
    # If a path is being parsed through the command line: load that config
    # Otherwise: default back to the experiment_parameters.yaml in the src/ directory
    print(f"Config used: {path_config}")
    if not path_config:
        path_config = cwd / "experiment_parameters.yaml"
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)

    # Load the database
    df_files = load_database(config)

    # Get the AudioProcessor parameters
    params_preprocessing = config["AudioProcessor"]["preprocessing"]
    params_vad = config["AudioProcessor"]["vad"]
    # params_augmentation = config["AudioProcessor"]["augmentation"]
    params_dataprocessing = config["DataProcessor"]
    params_modelling = config["ModelTrainer"]

    # High level config for splitting the data into train/dev/test
    cfg_meta_splitting = config["DataProcessor"]["meta"]

    # Get the path parameters
    cache_base_features = cwd / f"../{config['paths']['cache_features']}"
    cache_base_vad = cwd / f"../{config['paths']['cache_vad']}"
    cache_base_split = cwd / f"../{config['paths']['cache_split']}"

    # Get the keys of the keys in audio_processor_params
    keys_of_keys_preprocessing = [
        list(parameters.keys()) for parameters in params_preprocessing.values()
    ]
    # keys_of_keys_vad = [list(parameters.keys()) for parameters in params_vad.values()]

    # Pre-process the audio data successively and iterate over each combination of pre-processing step configurations
    for combination_preprocessing in itertools.product(*keys_of_keys_preprocessing):
        dct_preprocessing_parameters = {
            step: {key: params_preprocessing[step][key]}
            for step, key in zip(params_preprocessing.keys(), combination_preprocessing)
        }
        print(f"Current preprocessing parameters: {dct_preprocessing_parameters}")

        # Filter the database based on SNR and clipping - if desired
        df_files = filter_audio_quality(df_files, cwd, config)

        # Process the audio data with the current combination of parameters
        transforms = config_to_transforms(dct_preprocessing_parameters)
        df_files_preprocessed = apply_transforms(df_files, transforms)

        # Iterate over the different VAD methods
        for processing_type_vad, processing_params_vad in params_vad.items():
            print(f"Current VAD method: {processing_type_vad}")
            df_files_vad = apply_vad(
                df_files_preprocessed,
                processing_type_vad,
                processing_params_vad,
                cache_base_vad,
            )

            # Extract features
            idx_files_processed = df_files_vad.index
            for feature_set, extract_options in config["FeatureExtractor"][
                "feature_sets"
            ].items():
                df_features = extract_features(
                    idx_files_processed,
                    feature_set,
                    extract_options,
                    path_features=cache_base_features,
                )

                # Compose a string that describes the processing steps so far to store the indices of the data splits at
                # Term for the audio quality filtering applied
                filter_lists = config["database"]["filter_files"]["filter_lists"]
                filter_str = "filter-no_audio_quality_blacklist"
                if filter_lists:
                    filter_str = "filter-" + "-".join(filter_lists)
                str_experiment_data = compose_experiment_str(
                    [
                        filter_str,
                        dct_preprocessing_parameters,
                        {processing_type_vad: processing_params_vad},
                        {feature_set: extract_options},
                    ]
                )

                print(str_experiment_data)

                ### outer split ###
                # Iterate over the different configured outer splitting methods
                data_processor = DataProcessor(
                    df_features,
                    df_files_vad,
                    str_experiment_data,
                    cache_base_split,
                    cfg_meta_splitting,
                )
                # TODO: allocate splitting to outer CV class
                for outer_splitting, split_options in params_dataprocessing[
                    "splitting"
                ].items():
                    idx_train, idx_dev, idx_test = data_processor.load_and_process_data(
                        outer_splitting, split_options
                    )

                    ### Modelling - inner split ###
                    # Sanity check: print unique test speakers
                    print(f"Speakers in train partition:")
                    print(df_files_vad.loc[idx_test].participant_code.unique())
                    # Pack all meta parameter configs for the current run together for saving the config of the current run
                    lst_configs_run = [
                        # TODO: meta configs for FE and AP not accessed yet
                        {"database": config["database"]},
                        {"paths": config["paths"]},
                        {
                            "AudioProcessor": {
                                "preprocessing": dct_preprocessing_parameters,
                                "vad": {processing_type_vad: processing_params_vad},
                            },
                        },
                        {
                            "FeatureExtractor": {
                                "feature_sets": {feature_set: extract_options}
                            }
                        },
                        {
                            "DataProcessor": {
                                "meta": cfg_meta_splitting,
                                "splitting": {outer_splitting: split_options},
                            }
                        },
                    ]

                    # Initialize the ModelTrainer and train models with the processed data
                    # Integrate the experiment data string into the path for the results
                    str_experiment_data = f"{str_experiment_data}/{outer_splitting}"
                    print(str_experiment_data)
                    path_results_modelling = (
                        cwd / f"../{config['paths']['results_modelling']}"
                    )
                    model_trainer = ModelTrainer(
                        params_modelling,
                        df_files_vad,
                        df_features,
                        idx_train,
                        idx_dev,
                        idx_test,
                        str_experiment_data,
                        path_results_modelling,
                        path_config,
                        lst_configs_run,
                    )
                    model_trainer.run_experiment()


def load_database(config):
    """
    Load the database based on the provided configuration.
    Either a local database (csv file containing the labels) or an audb.Database can be loaded.

    Args:
        config (dict): The configuration dictionary containing the database parameters.

    Returns:
        pd.DataFrame: The DataFrame with the labels from the loaded database.
    """
    # Check which type of database is used
    # If the database is local: load the DataFrame with the labels
    if config["database"]["type"] == "local":
        # Load the DataFrame with the labels
        df_files = pd.read_csv(config["database"]["path_df_meta"])

        # Associate the absolute path to the data files
        path_data = config["database"]["path_data"]
        # If path_data is empty: assume that the file paths are already absolute
        if path_data:
            # Check if the concatenation of the file index and data path really points to some valid wav file in .iloc[0]
            path_test_wav = os.path.join(path_data, df_files.iloc[0]["file"])
            if os.path.isfile(path_test_wav) and path_test_wav.endswith(".wav"):
                # Associate the absolute path to the data files
                df_files["file"] = df_files["file"].apply(
                    lambda x: os.path.join(path_data, x)
                )
            else:
                raise ValueError(
                    f"The path {path_test_wav} does not point to a valid wav file."
                )

        # Set the index column
        index_column = config["database"]["index_column"]
        # Check if the index column entries are all valid columns in the DataFrame
        if index_column is not None and all(
            entry in df_files.columns for entry in index_column
        ):
            # If only one column is present: assume "file" and set it as the index
            if len(config["database"]["index_column"]) == 1:
                df_files = df_files.set_index(index_column[0])
            # Check if the common audformat index is present and convert it to the timedelta format
            elif all(entry in index_column for entry in ["file", "start", "end"]):
                # Convert the indices to the timedelta format
                df_files["start"] = df_files["start"].apply(pd.to_timedelta)
                df_files["end"] = df_files["end"].apply(pd.to_timedelta)
                # Restore the MultiIndex
                df_files.set_index(["file", "start", "end"], inplace=True)
            else:
                raise ValueError(f"The index column {index_column} is not supported.")
        else:
            raise ValueError(
                f"The index column {index_column} is either empty or not present in the DataFrame."
            )

        # Convert date-related columns into datetime format
        if "date" in df_files.columns:
            df_files["date"] = pd.to_datetime(df_files["date"])
        if "Date_audeering" in df_files.columns:
            df_files["Date_audeering"] = pd.to_datetime(df_files["Date_audeering"])

        # Filter out participants that have less than x sessions
        print(
            f"Number of participants before filtering out too few sessions: {len(df_files['participant_code'].unique())}; "
            f"Shape of the DataFrame before filtering: {df_files.shape}"
        )
        min_sessions = config["database"]["min_sessions"]
        unique_sessions_per_participant = df_files.groupby("participant_code")[
            "session"
        ].nunique()
        participants_too_few_sessions = unique_sessions_per_participant[
            unique_sessions_per_participant <= min_sessions
        ]
        list(participants_too_few_sessions.index)

        # Filter the main DataFrame
        df_files = df_files[
            ~df_files["participant_code"].isin(
                list(participants_too_few_sessions.index)
            )
        ]
        print(
            f"Number of participants after filtering out too few sessions: {len(df_files['participant_code'].unique())}; "
            f"Shape of the DataFrame after filtering: {df_files.shape}"
        )

        # Filter out particular prompts
        config["database"]["discard_prompts"]
        df_files = df_files[
            ~df_files.prompt.isin(config["database"]["discard_prompts"])
        ]
        print(f"Shape of the DataFrame after filtering prompts: {df_files.shape}")

        return df_files

    # If the database is audb: load the audb.Database
    elif config["database"]["type"] == "audb":
        # Get the database parameters
        db_name = config["database"]["name"]
        sampling_rate = config["database"]["sampling_rate"]
        version = config["database"]["version"]
        full_path = config["database"]["full_path"]
        cache_root = (
            audb.default_cache_root(shared=True)
            if config["database"]["shared_cache_root"]
            else None
        )
        bit_depth = config["database"]["bit_depth"]
        channels = config["database"]["channels"]
        format = config["database"]["format"]
        mixdown = config["database"]["mixdown"]

        # Load the database
        db = audb.load(
            db_name,
            sampling_rate=sampling_rate,
            version=version,
            full_path=full_path,
            cache_root=cache_root,
            bit_depth=bit_depth,
            channels=channels,
            format=format,
            mixdown=mixdown,
        )

        # Return the DataFrame with the labels
        df_files = db["files"].get()

    return df_files


def filter_audio_quality(df_files, cwd, config):
    """
    Filter the audio quality of files based on blacklists.
    The filter lists are generated in "notebooks/audio_quality_filtering.ipynb". If no config is found: continue without filtering if no filter criterion is set or throw error.

    Args:
        df_files (DataFrame): The DataFrame containing the files to be filtered.
        cwd (str): The current working directory.
        config (dict): The configuration settings.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        AssertionError: If the number of files in the filtered DataFrame does not match the expected number of files after filtering.
    """
    cfg_filter = config["database"]["filter_files"]
    path_blacklist_cfg = cfg_filter["path_blacklist"]
    cfg_blacklists = cfg_filter["filter_lists"]
    # Check if filtering should be performed to begin with and if the path to the blacklist config exists
    if path_blacklist_cfg is None or not os.path.isfile(
        os.path.join(cwd, path_blacklist_cfg)
    ):
        # Check if the list of blacklists is empty
        if not cfg_blacklists:
            warnings.warn(
                "No blacklists config file for audio-quality-based filtering provided, but the list of criteria to filter is also empty. Continuing without filtering."
            )
            return df_files
        # Raise error if filtering should be performed, but no path to the blacklist config is provided
        else:
            raise FileNotFoundError(
                f"The path in 'path_blacklist' does not exist: '{path_blacklist_cfg}'."
            )
    with open(os.path.join(cwd, path_blacklist_cfg), "r") as file:
        dct_filter_files = yaml.safe_load(file)

    # Get the current denoising setting to fetch the correct blacklist
    cur_denoising = list(config["AudioProcessor"]["preprocessing"]["denoising"].keys())[
        0
    ]

    # Select the set of blacklisted files based on the denoising approach used
    yaml_blacklists = dct_filter_files[cur_denoising]

    # Initialise the DataFrame to be returned - in case no filter is applied
    df_files_filtered = df_files.copy()
    if cfg_blacklists:
        print("Performing audio-quality-based filtering")
        len_all_duplicates = 0
        blacklists = set()
        for key in cfg_blacklists:
            len_cur_lst = 0
            if key in yaml_blacklists:
                for item in yaml_blacklists[key]:
                    if item in blacklists:
                        print(f"Duplicate item: {item}")
                        len_all_duplicates += 1
                    else:
                        blacklists.add(item)
                        len_cur_lst += 1
                print(f"Adding {key} with {len_cur_lst} samples.")
            else:
                warnings.warn(f"List {key} not found in the blacklists.")
        print(
            f"Filtered out {len(blacklists)} samples. The blacklists overlapped by {len_all_duplicates} duplicate sample(s)."
        )

        # Use the blacklists to filter the dataframe
        df_files_filtered = df_files[
            ~df_files.index.get_level_values("file").isin(blacklists)
        ]

        # Verify that the number of files in the filtered DataFrame matches the expected number of files after filtering
        assert len(df_files) - len(blacklists) == len(
            df_files_filtered
        ), "The number of files in the filtered DataFrame does not match the expected number of files after filtering."
    else:
        print(
            "No lists for audio-quality-based filtering provided. Continuing without filtering."
        )

    return df_files_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "path_config",
        type=str,
        nargs="?",
        default=None,
        help="Path to the configuration file",
    )

    args = parser.parse_args()

    main(args.path_config)
