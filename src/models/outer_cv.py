from collections import Counter
import numpy as np
import os
import pandas as pd
import pprint
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
import warnings

import audeer

from models.inner_cv import InnerCV
from models.process_results import Results


class OuterCV:
    """
    Class representing the outer cross-validation for modelling.

    Parameters:
    - parameters (tuple): Tuple containing the following parameters:
        - cohort (str): Cohort identifier.
        - speech_task (str): Speech task identifier.
        - feature_selection (dict): Feature selection configuration.
        - feature_normalization (str): Feature normalization method.
        - target_variable (str): Target variable name.
        - cv_config_outer (dict): Outer cross-validation configuration.
        - cv_config_inner (dict): Inner cross-validation configuration.
        - cv_method_inner (dict): Inner cross-validation method configuration.
        - estimator_config (dict): Estimator configuration.
    - cfg_meta (dict): Metadata configuration.
    - path_results_base (str): Base path for storing results.
    - path_config_experiment (str): Path to the experiment configuration file.
    - lst_configs_run (list): List of all meta parameters of the current run (for constructing the output path for the results).
    - df_files_filtered (pd.DataFrame): Filtered file metadata.
    - df_features_filtered (pd.DataFrame): Filtered feature data.
    - idx_train (pd.MultiIndex): Indices for the training partition.
    - idx_dev (pd.MultiIndex): Indices for the development partition.
    - idx_test (pd.MultiIndex): Indices for the test partition.
    """

    def __init__(
        self,
        parameters,
        cfg_meta,
        path_results_base,
        str_experiment_data,
        path_config_experiment,
        lst_configs_run,
        df_files_filtered,
        df_features_filtered,
        idx_train,
        idx_dev,
        idx_test,
    ):
        # Unpack and instantiate the parameters
        self.parameters = parameters
        (
            self.cohort,
            self.speech_task,
            self.feature_selection,
            self.feature_normalization,
            self.personalisation,
            self.target_variable,
            self.cv_config_outer,
            self.cv_config_inner,
            self.cv_method_inner,
            self.estimator_config,
        ) = self.parameters
        # Unpack the range of the target variable
        str_cur_target = None
        if type(self.target_variable) == dict:
            str_cur_target = list(self.target_variable.keys())[0]
            str_keys_target = self.target_variable.get(str_cur_target, {})
            if "range_raw" in str_keys_target and "range_normalized" in str_keys_target:
                self.target_variable_range_raw = self.target_variable[str_cur_target][
                    "range_raw"
                ]
                self.target_variable_range_normalized = self.target_variable[
                    str_cur_target
                ]["range_normalized"]
            else:
                raise KeyError(
                    "Dictionary as target variable passed, but the target variable range is not specified."
                )
        else:
            str_cur_target = self.target_variable
            self.target_variable_range_raw = None
            self.target_variable_range_normalized = None
        self.target_variable = str_cur_target
        # Unpack the CV configs
        # outer CV
        self.cv_strategy_outer = list(self.cv_config_outer.keys())[0]
        self.cv_settings_outer = self.cv_config_outer[self.cv_strategy_outer]
        # inner CV
        self.cv_strategy_inner = list(self.cv_config_inner.keys())[0]
        self.cv_settings_inner = self.cv_config_inner[self.cv_strategy_inner]
        # Gridsearch object for inner CV
        self.cv_inner_method_name = list(self.cv_method_inner.keys())[0]
        self.cv_inner_method_settings = self.cv_method_inner[self.cv_inner_method_name]

        self.cfg_meta = cfg_meta
        self.path_results_base = path_results_base
        self.str_experiment_data = str_experiment_data
        self.path_config_experiment = path_config_experiment
        self.lst_configs_run = lst_configs_run
        self.df_files_filtered = df_files_filtered
        self.df_features_filtered = df_features_filtered
        self.idx_train = idx_train
        self.idx_dev = idx_dev
        self.idx_test = idx_test
        # TODO: label encoding required?

    def setup_outer_cv(self):
        """
        Set up the outer cross-validation process.

        This method performs the following steps:
        1. Isolates the grouping column (e.g. speaker ID) from the metadata.
        2. Concatenates the train and dev indices for cross-validation.
        3. Selects and normalizes the features.
        """
        # Isolate the grouping column (e.g. speaker ID) from the metadata
        self._provide_groups()
        # Concatenate the train and dev indices for CV
        self._concat_train_dev_indices()
        # Target normalization for regression
        # To be done before feature selection, since that uses the target variable too
        self._normalize_target()

    def run_outer_cv(self):
        """
        Runs the outer cross-validation process.

        This method initializes the splitting objects for the outer cross-validation,
        creates the data splits, and performs the inner cross-validation for each fold.
        It tracks the results from all folds and stores the inner CV objects.
        """
        # Check if this modelling run was already completed in the past
        self.path_results = self._compile_output_directory()
        path_results_yaml = os.path.join(self.path_results, "models", "results.yaml")
        print(f"Performing modelling with the parameters:\n{self.path_results}")
        if os.path.exists(path_results_yaml) and os.path.getsize(path_results_yaml) > 0:
            print(
                f"\nThis modelling run was already completed in the past.\nSKIPPING\n"
            )
            return "existing"

        # Initialize the splitting objects for the outer cross validation
        self._initialize_outer_splitting()

        # Create the data splits for the outer cross validation
        iterator_split_outer = self.outer_cv.split(
            X=self.df_features_filtered,
            y=self.df_files_filtered[self.target_variable],
            groups=self.groups[self.cfg_meta["groups"]],
        )

        # Fold index for tracking the results
        idx_fold = 0
        # DataFrame to track results from all folds
        self.df_results_outer_test = pd.DataFrame()
        self.df_results_outer_train = pd.DataFrame()
        # Dictionary to store the inner CV objects to
        self.dct_inner_cv_objects = {}

        # Print the parameters of the current run
        # For tmux: need to manually print each line separately, since \n is not getting parsed
        # Import here to avoid circular import error
        from utils import map_modelling_config, print_line_by_line

        print_line_by_line(
            pprint.pformat(map_modelling_config(self.parameters)["ModelTrainer"])
        )

        for train_index, test_index in audeer.progress_bar(
            iterable=iterator_split_outer,
            total=self.outer_cv.get_n_splits(
                self.df_features_filtered,
                self.df_files_filtered[self.target_variable],
                self.groups[self.cfg_meta["groups"]],
            ),
            desc=f"Outer CV fold; {self.outer_cv}",
        ):
            # If personalisation is to be performed:
            # Move samples from the training set to the test set here
            if list(self.personalisation.keys())[0] == "none":
                print("No personalisation")
                pass
            else:
                print(f"Performing the personalisation {self.personalisation}")
                train_index, test_index = self._apply_personalisation(
                    train_index, test_index
                )

            # Normalise the features depending on the ACTUAL train and test indices
            # (Before: train/test indices of fixed test set were used)
            # Select and normalize the features
            self._select_features(train_index)
            self._normalize_features(train_index)

            # Initialize the InnerCV class with the relevant parameters
            inner_cv = InnerCV(
                train_index,
                test_index,
                self.df_features_filtered,
                self.df_files_filtered[self.target_variable],
                self.groups,
                self.cfg_meta,
                self.target_variable,
                self.cv_strategy_inner,
                self.cv_inner_method_name,
                self.cv_settings_inner,
                self.cv_inner_method_settings,
                self.estimator_config,
                idx_fold,
            )
            # Use the InnerCV class to fit the data
            inner_cv.run_inner_cv()
            (
                df_results_inner_train,
                df_results_inner_test,
                inner_cv_object,
            ) = inner_cv.process_results()

            self.df_results_outer_test = pd.concat(
                [self.df_results_outer_test, df_results_inner_test]
            )
            self.df_results_outer_train = pd.concat(
                [self.df_results_outer_train, df_results_inner_train]
            )

            self._max_voting(self.df_results_outer_test)

            # Store the inner CV object
            self.dct_inner_cv_objects[idx_fold] = inner_cv_object

            idx_fold += 1

        self.compile_results()

        return None

    def compile_results(self):
        """
        Compiles the results of the outer cross-validation.
        This is done by initialising the Results class and calling its compile_results method.
        """
        # Compile the results
        results = Results(
            self.cfg_meta,
            self.df_features_filtered,
            self.df_files_filtered,
            self.groups,
            self.target_variable,
            self.df_results_outer_test,
            self.df_results_outer_train,
            self.df_results_maxvote,
            self.dct_inner_cv_objects,
            self.estimator_config,
        )

        results.compile_results(
            self.path_results,
            self.path_config_experiment,
            self.lst_configs_run,
            self.parameters,
        )

    def _compile_output_directory(self):
        """
        Compile the output directory for storing the results of the experiment.

        Returns:
            str: The path to the output directory.
        """
        # Hack to re-use composition of string for output directory
        from main import compose_experiment_str

        # Get the directory of this script to compose the relative path to the results directory

        # Compose target variable handling: depending on normalisation
        def compose_target_variable_string(self):
            parts = []
            if self.target_variable_range_raw is not None:
                # Tuples are not supported by YAML: use lists
                assert len(self.target_variable_range_raw) == 2
                parts.append(
                    f"raw_{self.target_variable_range_raw[0]}_{self.target_variable_range_raw[1]}"
                )
            if self.target_variable_range_normalized is not None:
                assert len(self.target_variable_range_normalized) == 2
                parts.append(
                    f"normalized_{self.target_variable_range_normalized[0]}_{self.target_variable_range_normalized[1]}"
                )
            # If both are None: no target normalisation was performed and the subdirectory should indicate so
            if len(parts) == 0:
                parts = ["raw_target_values"]
            return os.path.join(self.target_variable, "-".join(parts))

        str_target_variable = compose_target_variable_string(self)

        # Compile the output directory
        feature_selection = self.feature_selection
        if self.feature_selection["type"] == "manual_feature_selection":
            feature_selection = {
                self.feature_selection["type"]: self.feature_selection["description"]
            }
        dir_results = compose_experiment_str(
            [
                # Base target variable - irregardless of normalisation
                str_target_variable,
                # Putting full list of prompts is too long
                # --> check experiment_parameters.yaml for exact mapping and use only the identifier key here
                {"speechtasks": list(self.speech_task.keys())[0]},
                {"cohort": self.cohort},
                self.str_experiment_data,
                feature_selection,
                self.feature_normalization,
                {"personalisation": list(self.personalisation.keys())[0]},
                self.cv_config_outer,
                self.cv_config_inner,
                self.cv_method_inner,
                {
                    self.estimator_config["type"]: self.estimator_config[
                        "grid_description"
                    ]
                },
            ]
        )
        # Fuse with the base path and create the directory
        return audeer.mkdir(self.path_results_base / dir_results)

    def _apply_personalisation(self, train_index, test_index):
        # Move samples from the training set to the test set
        # based on the personalisation configuration

        # Get the surveys that are used for personalisation
        personalisation_surveys = self.personalisation[
            list(self.personalisation.keys())[0]
        ]["surveys"]
        personalisation_rows = self.df_files_filtered["survey"].isin(
            personalisation_surveys
        )
        # Map the train and test indices back to the label DataFrame
        # and get the rows that are used for personalisation
        df_labels_test = self.df_files_filtered.iloc[test_index]
        df_labels_test = df_labels_test[
            df_labels_test["survey"].isin(personalisation_surveys)
        ]

        # Get the numeric indices of df_labels_test in self.df_files_filtered
        personalisation_indices = self.df_files_filtered.index.get_indexer(
            df_labels_test.index
        )

        # Convert train_index and test_index to sets for efficient operations
        train_index_set = set(train_index)
        test_index_set = set(test_index)

        # Remove the personalisation indices from test_index
        test_index_set -= set(personalisation_indices)

        # Add the personalisation indices to train_index
        train_index_set |= set(personalisation_indices)

        # Convert the sets back to lists
        modified_train_index = np.array(list(train_index_set))
        modified_test_index = np.array(list(test_index_set))

        print(
            f"Personalisation:\n"
            f"Samples to be moved: {len(personalisation_indices)}\n"
            f"len() train_index before moving: {len(train_index)}; after moving: {len(modified_train_index)}\n"
            f"len() test_index before moving: {len(test_index)}; after moving: {len(modified_test_index)}"
        )
        if len(train_index) + len(personalisation_indices) != len(modified_train_index):
            warnings.warn(
                f"len(train_index) + len(modified_train_index): {len(train_index) + len(personalisation_indices)}\n"
                f"len(modified_train_index): {len(modified_train_index)}"
                f"The lengths should be the same."
            )
        if len(test_index) - len(personalisation_indices) != len(modified_test_index):
            warnings.warn(
                f"len(test_index) - len(modified_test_index): {len(test_index) - len(personalisation_indices)}\n"
                f"len(modified_test_index): {len(modified_test_index)}"
                f"The lengths should be the same."
            )

        return modified_train_index, modified_test_index

    def _max_voting(self, df_results_test):
        """
        Perform max voting to determine the majority label for each group and session.
        Instantiates the DataFrame df_results_maxvote with the results of the max voting.

        Args:
            df_results_test (DataFrame): The DataFrame containing the test results.
        """
        # Construct DataFrame on "group" and potentially "session" level to do max voting across all respectively aggregated recordings
        lst_group_by_columns = [self.cfg_meta["groups"]]
        # Optional: column to further group by individual sessions
        if self.cfg_meta["sessions"] is not None:
            lst_group_by_columns.append(self.cfg_meta["sessions"])
        # Get a series with the true target labels per speaker
        srs_y = df_results_test.groupby(lst_group_by_columns)[
            self.target_variable
        ].unique()

        # Double-check that only unique labels per group and session are present
        assert (
            len(set(map(len, list(srs_y)))) == 1
        ), "The class labels per speaker have to be unique!"

        # Construct a series of arrays from the predictions and use
        # Counter.most_common to retrieve the max voting label
        srs_preds = df_results_test.groupby(lst_group_by_columns)["predictions"].apply(
            list
        )
        # Get the majority label from all predictions per speaker as a series
        srs_preds = srs_preds.apply(lambda row: Counter(row).most_common()[0][0])
        # Compose the DataFrame for the majority voting predictions
        self.df_results_maxvote = pd.concat(
            [srs_y.apply(pd.Series), srs_preds.to_frame()], axis=1
        )
        # Correct the column name of the true labels
        self.df_results_maxvote.rename(columns={0: self.target_variable}, inplace=True)

    def _provide_groups(self):
        """
        Filters the label DataFrame for the groups and, if provided, sessions columns.
        Instantiates the following attributes:
        - groups (DataFrame): The groups DataFrame.
        - groups_train (DataFrame): The groups DataFrame for the training partition.
        - groups_dev (DataFrame): The groups DataFrame for the development partition.
        - groups_test (DataFrame): The groups DataFrame for the test partition.
        """
        # List with the columns to filter out
        lst_cols_grouping = [self.cfg_meta["groups"]]
        # Optional: column to further group by individual sessions
        if self.cfg_meta["sessions"] is not None:
            lst_cols_grouping.append(self.cfg_meta["sessions"])
        # Combine the grouping columns to a single column for the inner CV splitting method
        self.df_files_filtered["group"] = self.df_files_filtered[
            lst_cols_grouping
        ].apply(lambda x: "_".join(x.astype(str)), axis=1)
        # Add the group column to the list of columns to filter, since it will be utilized for the inner CV splitting method
        # If only "groups" is given: the content of "groups" will simply be copied to the "group" column
        lst_cols_grouping.append("group")
        # Filter the label DataFrame for the groups and, if provided, sessions columns
        self.groups = self.df_files_filtered[lst_cols_grouping]
        self.groups_train = self.df_files_filtered.loc[
            self.idx_train, lst_cols_grouping
        ]
        # Check if there is a development partition
        if not self.idx_dev.empty:
            self.groups_dev = self.df_files_filtered.loc[
                self.idx_dev, lst_cols_grouping
            ]
        else:
            self.groups_dev = pd.Series()
        self.groups_test = self.df_files_filtered.loc[self.idx_test, lst_cols_grouping]

    def _concat_train_dev_indices(self):
        """
        Concatenates the train and dev indices for cross-validation.

        If the dev indices are not empty, the train and dev indices are concatenated
        along with their respective groups. Otherwise, only the train indices and
        groups are used.
        """
        if not self.idx_dev.empty:
            self.idx_train_dev = self.idx_train.append(self.idx_dev)
            # TODO: test this
            self.groups_train_dev = self.groups_train.append(self.groups_dev)
        else:
            self.idx_train_dev = self.idx_train
            self.groups_train_dev = self.groups_train

    def _select_features(self, train_index):
        """
        Implements and performs different feature selection methods.
        Only the train and dev partitions are used to fit the feature selection.

        Raises:
            ValueError: If an unknown feature selection type is specified.
        """
        # Select the features
        # Use only train and dev to fit the feature selection
        self.feature_selector = None
        if self.feature_selection["type"] == "no_feature_selection":
            return
        elif self.feature_selection["type"] == "logistic_regression":
            self.feature_selector = SelectFromModel(
                LogisticRegression(
                    max_iter=self.feature_selection["max_iter"],
                    n_jobs=self.cfg_meta["num_workers"],
                )
            )
            self.feature_selector.fit(
                self.df_features_filtered.iloc[train_index],
                self.df_files_filtered[self.target_variable].loc[train_index],
            )
            # Remove the features that are not selected
            self.df_features_filtered = self.df_features_filtered.loc[
                :, self.feature_selector.get_support()
            ]
        elif self.feature_selection["type"] == "manual_feature_selection":
            print(
                f"Performing manual feature selection\n"
                f"Length before removal: {len(self.df_features_filtered.columns)} features"
            )
            # Either or: include or omit features
            if (
                "lst_include" in self.feature_selection
                and not "lst_omit" in self.feature_selection
            ):
                lst_include = self.feature_selection["lst_include"]
                # Validate lst_include: are all provided features really in the DataFrame?
                invalid_features = [
                    feature
                    for feature in lst_include
                    if feature not in self.df_features_filtered.columns
                ]
                if invalid_features:
                    raise ValueError(
                        f"The following features in lst_include are not valid columns: {invalid_features}"
                    )
                # Filter the DataFrame
                self.df_features_filtered = self.df_features_filtered[lst_include]

            # In case also a list to omit features is given:
            elif (
                "lst_omit" in self.feature_selection
                and not "lst_include" in self.feature_selection
            ):
                lst_omit = self.feature_selection["lst_omit"]

                # Validate lst_include: are all provided features really in the DataFrame?
                invalid_features = [
                    feature
                    for feature in lst_omit
                    if feature not in self.df_features_filtered.columns
                ]
                if invalid_features:
                    raise ValueError(
                        f"The following features in lst_include are not valid columns: {invalid_features}"
                    )
                # Filter the DataFrame to exclude features to be omitted
                self.df_features_filtered = self.df_features_filtered.drop(
                    columns=lst_omit
                )
            else:
                raise ValueError(
                    "Manual feature selection requires either a list of features to include or to omit."
                )
            print(
                f"Length after removal: {len(self.df_features_filtered.columns)} features"
            )
        else:
            raise ValueError(
                f"Unknown feature selection type: {self.feature_selection['type']}"
            )

    def _normalize_features(self, train_index):
        """
        Implements and performs different feature normalization methods.
        Only the train and dev partitions are used to fit the feature normalization.

        Raises:
            ValueError: If an unknown normalization type is specified.
        """
        # Normalize the features
        if self.feature_normalization == "sklearn_standard_scaler":
            scaler = StandardScaler()
            scaler.fit(self.df_features_filtered.iloc[train_index])
            self.df_features_filtered = pd.DataFrame(
                scaler.transform(self.df_features_filtered),
                index=self.df_features_filtered.index,
                columns=self.df_features_filtered.columns,
            )
        elif self.feature_normalization == "z_score_normalization":
            # TODO: check if ChatGPT is right on this
            mean = self.df_features_filtered.iloc[train_index].mean()
            std = self.df_features_filtered.iloc[train_index].std()
            self.df_features_filtered = (self.df_features_filtered - mean) / std
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalization['type']}"
            )

    def _normalize_target(self):
        """ """
        # Check if both range lists are not None
        if (
            self.target_variable_range_raw is not None
            and self.target_variable_range_normalized is not None
        ):
            print(
                f"Performing target normalization:\n"
                f"Range before normalisation (min - max data): {self.df_files_filtered[self.target_variable].min()} - "
                f"{self.df_files_filtered[self.target_variable].max()}; "
                f"full scale: {str(self.target_variable_range_raw)}"
            )
            # Extract the raw and normalized ranges
            raw_min, raw_max = self.target_variable_range_raw
            norm_min, norm_max = self.target_variable_range_normalized

            # Normalize the target variable
            self.df_files_filtered.loc[:, self.target_variable] = (
                (self.df_files_filtered[self.target_variable] - raw_min)
                / (raw_max - raw_min)
            ) * (norm_max - norm_min) + norm_min
            print(
                f"Range after normalisation (min - max data): {self.df_files_filtered[self.target_variable].min()} - "
                f"{self.df_files_filtered[self.target_variable].max()}; "
                f"full scale for normalisation: {str(self.target_variable_range_normalized)}"
            )
        elif (
            self.target_variable_range_raw is None
            or self.target_variable_range_normalized is None
        ):
            # Raise a warning if only one of the two lists is None
            warnings.warn(
                "One of the target variable range lists is None. Normalization will not be performed."
            )

    def _initialize_outer_splitting(self):
        """
        Initializes the outer cross validation splitting object based on the specified CV strategy.

        Raises:
            NotImplementedError: If the CV strategy is "fixed_test_set" and fixed speaker lists are not implemented yet.
            ValueError: If the CV strategy is unknown.
        """
        # Initialise the outer cross validation splitting object
        if self.cv_strategy_outer == "fixed_test_set":
            # Manually define one outer split based on either fixed indices for the test partition or a list of speakers
            if self.idx_train.empty and self.idx_test.empty:
                raise NotImplementedError("Fixed speaker lists not implemented yet.")
            else:
                self.outer_cv = DefinedTestSet(
                    flag_indices_or_speakers="indices", test_partition=self.idx_test
                )
        elif self.cv_strategy_outer == "k_fold":
            self.outer_cv = KFold(
                n_splits=self.cv_settings_inner["n_splits"],
                shuffle=self.cv_settings_inner["shuffle"],
                random_state=(
                    self.cv_settings_inner["random_state"]
                    if self.cv_settings_inner["shuffle"]
                    else None
                ),
            )
        elif self.cv_strategy_outer == "group_k_fold":
            self.outer_cv = GroupKFold(
                n_splits=self.cv_settings_inner["n_splits"],
            )
        elif self.cv_strategy_outer == "stratified_group_k_fold":
            self.outer_cv = StratifiedGroupKFold(
                n_splits=self.cv_settings_inner["n_splits"],
                shuffle=self.cv_settings_inner["shuffle"],
                random_state=self.cv_settings_inner["random_state"],
            )
        elif self.cv_strategy_outer == "loso":
            self.outer_cv = LeaveOneGroupOut()
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy_outer}")


class DefinedTestSet:
    """
    Custom cross-validation splitter that defines a fixed test set.

    Parameters:
    - flag_indices_or_speakers (str): Indicates whether the test set is defined by indices or a list of speakers.
    - test_partition (list or pd.Index): The indices or speakers to be used as the test partition.

    Methods:
    - get_n_splits(X, y, groups): Returns the number of splits (always 1 in this case).
    - split(X, y, groups): Splits the data into train and test sets based on the defined test set.
    """

    def __init__(self, flag_indices_or_speakers, test_partition):
        self.flag_indices_or_speakers = flag_indices_or_speakers
        self.test_partition = test_partition

    def get_n_splits(self, X, y, groups):
        return 1

    def split(self, X, y, groups):
        """
        Splits the data into train and test sets based on the defined test set.

        Parameters:
        - X: The input features.
        - y: The target variable.
        - groups: The grouping variable.

        Returns:
        - List of tuples: Each tuple contains the indices of the train and test sets.
        """
        # Reset file-based index (since sklearn wants a numeric
        # one)
        groups_num = groups.reset_index()
        if self.flag_indices_or_speakers == "indices":
            # Get the numeric indices of the test partition
            test_indices = X.index.get_indexer(self.test_partition)

            # Get all indices
            all_indices = np.arange(X.shape[0])

            # Get the indices that are not in the test partition
            train_indices = np.setdiff1d(all_indices, test_indices)

            return [(train_indices, test_indices)]

        elif self.flag_indices_or_speakers == "speakers":
            # Get the numerical indices of the data frame that match
            # with the speakers to be excluded as a test partition
            test_index = groups_num[groups_num.speaker.isin(self.test_partition)].index
            # The inverse (~) of the test speakers is the train/
            # validation partition
            train_index = groups_num[
                ~groups_num.speaker.isin(self.self.test_partition)
            ].index
            return [(train_index, test_index)]
        else:
            raise ValueError(
                f"Unknown flag for test partition: {self.flag_indices_or_speakers}"
            )
