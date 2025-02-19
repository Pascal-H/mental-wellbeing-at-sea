import pandas as pd
from pathlib import Path
import numpy as np
import warnings

import audeer

import splitutils


class DataProcessor:
    """
    Class for processing data and generating train, dev, and test splits.

    Args:
        df_features (pd.DataFrame): The features DataFrame.
        df_files (pd.DataFrame): The files DataFrame.
        str_experiment_data (str): The experiment data string.
        cache_base (str): The base cache directory.
        cfg_meta (dict): The meta configuration.

    Methods:
        load_and_process_data: Loads and processes the data, generating train, dev, and test splits.
        _save_idx: Saves the train, dev, and test indices to disk.
        _load_split_idx: Loads the train, dev, and test indices from disk.
        _splitutils_traintest: Performs the traintest split using splitutils.
    """

    def __init__(
        self, df_features, df_files, str_experiment_data, cache_base, cfg_meta
    ):
        self.df_features = df_features
        self.df_files = df_files
        self.str_experiment_data = str_experiment_data
        self.cache_base = cache_base
        self.cfg_meta = cfg_meta

    def load_and_process_data(self, outer_splitting, split_options):
        """
        Loads and processes the data, generating train, dev, and test splits.

        Args:
            outer_splitting (str): The outer splitting method.
            split_options (dict): The split options.

        Returns:
            tuple: A tuple containing the train, dev, and test indices.

        Raises:
            ValueError: If the indices are empty.
        """
        # TODO: Filter data before splitting?
        # Get UID from split config to encode path
        path_outer_split = audeer.mkdir(
            self.cache_base / self.str_experiment_data / outer_splitting
        )
        cache_cur_split = audeer.mkdir(
            Path(path_outer_split)
            / audeer.uid(from_string=str(list(self.df_files.index)), short=True)
        )

        # Check if there are already indices for the current split setting
        flag_splits_exist = False
        files_exist = {
            f: (Path(cache_cur_split) / f).exists()
            for f in ["train.csv", "dev.csv", "test.csv"]
        }
        if all(files_exist.values()):
            flag_splits_exist = True

        # If desired: try to load the indices from disk
        if self.cfg_meta["try_load_existing_split"] == True:
            if flag_splits_exist:
                idx_train, idx_dev, idx_test = self._load_split_idx(cache_cur_split)
                return idx_train, idx_dev, idx_test
            else:
                warnings.warn(
                    f"No indices for the current split setting found in {cache_cur_split}. "
                    f"This contradicts the config with 'try_load_existing_split' = {self.cfg_meta['try_load_existing_split']}; splits are being calculated anew."
                )
        else:
            if flag_splits_exist:
                warnings.warn(
                    f"Indices for the current split setting found in {cache_cur_split}. "
                    f"This contradicts the config with 'try_load_existing_split' = {self.cfg_meta['try_load_existing_split']}; splits are being calculated anew."
                )

        if outer_splitting == "splitutils_traintest":
            idx_train, idx_dev, idx_test, goodness = self._splitutils_traintest(
                split_options
            )
        elif outer_splitting == "fixed_test_speakers":
            idx_train, idx_dev, idx_test = self._fixed_test_speakers(split_options)
            return idx_train, idx_dev, idx_test
        else:
            raise ValueError(f"Unknown outer splitting method: {outer_splitting}")

        self._save_idx(idx_train, idx_dev, idx_test, cache_cur_split, goodness)

        return idx_train, idx_dev, idx_test

    def _fixed_test_speakers(self, split_options):
        """
        Use a list of fixed speakers for the test set.
        """
        idx_test = self.df_files[
            self.df_files[split_options["speaker_column"]].isin(
                split_options["test_speakers"]
            )
        ].index
        idx_train = self.df_files[
            ~self.df_files[split_options["speaker_column"]].isin(
                split_options["test_speakers"]
            )
        ].index
        idx_dev = pd.DataFrame()

        assert len(idx_train) + len(idx_test) == len(self.df_features)

        return idx_train, idx_dev, idx_test

    def _save_idx(self, idx_train, idx_dev, idx_test, path_cache, goodness=None):
        """
        Saves the train, dev, and test indices to disk.

        Args:
            idx_train (pd.Index): The train indices.
            idx_dev (pd.Index): The dev indices.
            idx_test (pd.Index): The test indices.
            path_cache (str): The path to the cache directory.

        Raises:
            ValueError: If all indices are empty.
        """
        # Create a dictionary to map file names to data
        data_dict = {"train.csv": idx_train, "dev.csv": idx_dev, "test.csv": idx_test}

        # Check if all of the idx objects are empty
        if all(data.empty for data in data_dict.values()):
            # If all idx objects are empty, raise an error
            raise ValueError("All idx objects are empty. Nothing to save.")

        # Iterate over the dictionary and save each DataFrame to a .csv file
        for filename, data in data_dict.items():
            if data.empty:
                # If it's empty, create an empty DataFrame and save that
                pd.DataFrame().to_csv(Path(path_cache) / filename)
            else:
                df_idx = data.to_frame()
                df_idx.reset_index(drop=True, inplace=True)
                df_idx.to_csv(Path(path_cache) / filename, index=False)

        # If existing: save the goodness of the split dictionary to a file
        if goodness is not None:
            with open(Path(path_cache) / "goodness.txt", "w") as file:
                file.write(str(goodness))

    def _load_split_idx(self, path_cache):
        """
        Loads the train, dev, and test indices from disk.

        Args:
            path_cache (str): The path to the cache directory.

        Returns:
            tuple: A tuple containing the train, dev, and test indices.
        """
        # Create a dictionary to map file names to data
        data_dict = {}

        # Iterate over the file names and load each .csv file
        for filename in ["train.csv", "dev.csv", "test.csv"]:
            # Load the DataFrame from the .csv file
            idx = pd.read_csv(Path(path_cache) / filename)

            if not idx.empty:
                # Convert the indices to the timedelta format
                idx["start"] = idx["start"].apply(pd.to_timedelta)
                idx["end"] = idx["end"].apply(pd.to_timedelta)
                # Set the multi-index respectively
                # idx.set_index(["file", "start", "end"], inplace=True)

                # Restore the MultiIndex
                idx = pd.MultiIndex.from_frame(idx)

            # Add the DataFrame to the dictionary
            data_dict[filename] = idx

        return data_dict["train.csv"], data_dict["dev.csv"], data_dict["test.csv"]

    def _splitutils_traintest(self, split_options):
        """
        Performs the traintest split using splitutils.

        Args:
            split_options (dict): The split options.

        Returns:
            tuple: A tuple containing the train, dev, and test indices.
        """
        X = self.df_features
        # For now: high redundancy to convert all NaN occurrences to string for splitutils to work
        y = self.df_files[split_options["target"]].to_numpy()
        # Only possible if the target is a string
        if y.dtype.kind in ["S", "U", "O"]:
            y = np.nan_to_num(y, nan="NaN").astype(str)
        if split_options["variable_type"] == "continuous":
            # Float indicates continuous variable for splitutils
            y = y.astype(float)
        split_on = self.df_files[split_options["split_on"]].to_numpy()
        split_on = np.nan_to_num(split_on, nan="NaN").astype(str)
        # Unpack the stratification variables from the config
        stratif_cols = split_options["stratif_cols"]
        # dct_stratif = {var: self.df_files[var].to_numpy() for var in stratif_cols}
        # Quick hack for age: overwrite the stratification variable with binned age
        dct_stratif = {
            var: np.nan_to_num(self.df_files[var].to_numpy(), nan="NaN").astype(str)
            for var in stratif_cols
        }
        if "age" in stratif_cols:
            dct_stratif["age"] = splitutils.binning(self.df_files["age"], nbins=3)
        if "stress_current" in stratif_cols:
            dct_stratif["stress_current"] = splitutils.binning(
                self.df_files["stress_current"], nbins=10
            )

        # FIND OPTIMAL SPLIT
        train_i, test_i, goodness = splitutils.optimize_traintest_split(
            X=X,
            y=y,
            split_on=split_on,
            stratify_on=dct_stratif,
            weight=split_options["weights"],
            test_size=split_options["test_size"],
            k=split_options["k"],
            seed=split_options["seed"],
        )

        # Map the array indices back to the original dataframe indices
        idx_train = self.df_features.iloc[train_i].index
        idx_test = self.df_features.iloc[test_i].index
        # idx_dev should be empty for this mode
        idx_dev = pd.DataFrame()

        assert len(idx_train) + len(idx_test) == len(self.df_features)

        # Output the results
        # TODO: potentially save the text output for goodness of the split to a file
        print("goodness of split:\n", goodness)

        return idx_train, idx_dev, idx_test, goodness
