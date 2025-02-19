import itertools
import pandas as pd
import warnings

from models.outer_cv import OuterCV


class ModelTrainer:
    """
    Class for running experiments and training models.

    Args:
        config_modelling (dict): Configuration for the modelling setup.
        df_files (pd.DataFrame): DataFrame containing file metadata.
        df_features (pd.DataFrame): DataFrame containing feature data.
        idx_train (pd.MultiIndex): Indices for the training partition.
        idx_dev (pd.MultiIndex): Indices for the development partition.
        idx_test (pd.MultiIndex): Indices for the test partition.
        str_experiment_data (str): Experiment data identifier.
        path_results_base (str): Base path for storing experiment results.
        path_config_experiment (str): Path to the experiment configuration file.
        lst_configs_run (list): List of all meta parameters of the current run (for constructing the output path for the results).

    Attributes:
        config_modelling (dict): Configuration for the modelling setup.
        df_files (pd.DataFrame): DataFrame containing file metadata.
        df_features (pd.DataFrame): DataFrame containing feature data.
        idx_train (pd.MultiIndex): Indices for the training partition.
        idx_dev (pd.MultiIndex): Indices for the development partition.
        idx_test (pd.MultiIndex): Indices for the test partition.
        str_experiment_data (str): Experiment data identifier.
        path_results_base (str): Base path for storing experiment results.
        path_config_experiment (str): Path to the experiment configuration file.
        lst_configs_run (list): List of all meta parameters of the current run (for constructing the output path for the results).
        cfg_meta (list): Configuration for the experiment meta parameters.
        cfg_cohort (list): List of configurations for the filtering the cohort.
        cfg_speech_tasks (list): List of configurations for filtering the speech task.
        cfg_feature_selections (list): List of configurations for feature selection.
        cfg_feature_normalizations (list): List of configurations for feature normalization.
        cfg_target_variables (list): List of target variables to iterate over.
        cfg_cv_strategies_outer (list): List of configurations for the outer cross-validation strategies.
        cfg_cv_strategies_inner (list): List of configurations for the inner cross-validation strategies.
        cv_methods_inner (list): List of configurations for the inner cross-validation methods.
        cfg_estimators (list): List of configurations for the estimators.
    """

    def __init__(
        self,
        config_modelling,
        df_files,
        df_features,
        idx_train,
        idx_dev,
        idx_test,
        str_experiment_data,
        path_results_base,
        path_config_experiment,
        lst_configs_run,
    ):
        self.config_modelling = config_modelling
        self.df_files = df_files
        self.df_features = df_features
        self.idx_train = idx_train
        self.idx_dev = idx_dev
        self.idx_test = idx_test
        self.str_experiment_data = str_experiment_data
        self.path_results_base = path_results_base
        # Get the whole experiment config path to copy the config file to the results directory
        self.path_config_experiment = path_config_experiment
        self.lst_configs_run = lst_configs_run

        # Split the main config into sub configs
        self.cfg_meta = config_modelling["meta"]
        self.cfg_cohort = config_modelling["cohorts"]
        self.cfg_speech_tasks = config_modelling["speech_tasks"]
        self.cfg_feature_selections = config_modelling["feature_selections"]
        self.cfg_feature_normalizations = config_modelling["feature_normalizations"]
        self.cfg_personalisations = config_modelling["personalisations"]
        self.cfg_target_variables = config_modelling["target_variables"]
        self.cfg_cv_strategies_outer = config_modelling["cv_strategies_outer"]
        self.cfg_cv_strategies_inner = config_modelling["cv_strategies_inner"]
        self.cv_methods_inner = config_modelling["cv_methods_inner"]
        self.cfg_estimators = config_modelling["estimators"]

    def run_experiment(self):
        """
        Run the experiment by iterating over different combinations of parameters and performing outer cross-validation.
        """
        # Serialize the different parameters of the different sub-configs to create an iterable over the whole modelling setup
        configs = [
            self.cfg_cohort,
            self.cfg_speech_tasks,
            self.cfg_feature_selections,
            self.cfg_feature_normalizations,
            self.cfg_personalisations,
            self.cfg_target_variables,
            self.cfg_cv_strategies_outer,
            self.cfg_cv_strategies_inner,
            self.cv_methods_inner,
            self.cfg_estimators,
        ]

        # Use itertools.product to generate all combinations of parameters
        for combination in itertools.product(*configs):
            # Initialise the outer cross validation for the current experiment parameters
            outer_cv = OuterCV(
                combination,
                self.cfg_meta,
                self.path_results_base,
                self.str_experiment_data,
                self.path_config_experiment,
                self.lst_configs_run,
                df_files_filtered=None,
                df_features_filtered=None,
                idx_train=None,
                idx_dev=None,
                idx_test=None,
            )

            # Filter the metadata and features based on the config
            self._filter_data(
                outer_cv.cohort, outer_cv.speech_task, outer_cv.target_variable
            )
            # Pass the filtered data to the model
            outer_cv.df_files_filtered = self.df_files_filtered
            outer_cv.df_features_filtered = self.df_features_filtered
            # TODO: allocate splitting to outer CV class
            outer_cv.idx_train = self.idx_train_filtered
            outer_cv.idx_dev = self.idx_dev_filtered
            outer_cv.idx_test = self.idx_test_filtered

            # Lay out the data for the outer cross validation
            outer_cv.setup_outer_cv()

            # Perform the outer cross validation
            # Get string existing" back, if the results already exist and they don't need to be re-calculated
            flag_existing = outer_cv.run_outer_cv()
            # Compile the results
            if not flag_existing:
                outer_cv.compile_results()

    def _filter_data(self, cohort, speech_task, target_variable):
        """
        Filter the metadata and features based on the configuration.

        Args:
            cohort (str): Cohort to filter for.
            speech_task (dict): Dictionary speech task mapping to include only specific prompts.
            target_variable (str): Target variable to filter for.

        Returns:
            tuple: A tuple containing the filtered DataFrame for file metadata and feature data.

        Raises:
            ValueError: If the cohort or speech task is unknown.
        """
        # Have to re-initialise the filtered DataFrames and indices for each iteration
        self.df_files_filtered = self.df_files
        self.df_features_filtered = self.df_features
        self.idx_train_filtered = self.idx_train
        self.idx_dev_filtered = self.idx_dev
        self.idx_test_filtered = self.idx_test

        # Filter the metadata and features, as well as split indices, based on the config
        # Remove rows that contain only NaN values for the features
        # Find rows that contain only NaN values for the features
        rows_filter_features = self.df_features_filtered[
            self.df_features_filtered.isna().all(axis=1)
        ]
        rows_filter_targets = self.df_files_filtered[
            self.df_files_filtered[target_variable].isna()
        ]
        print(f"Filter DataFrames and indices for speech tasks: {speech_task}")
        self._remove_rows_df_idx(rows_filter_features, rows_filter_targets)

        if cohort == "all":
            pass
        elif type(cohort) == dict:
            # Provide correct format to filter for the cohort columns
            lst_cohort = []
            # cohort can be a list with 1 or more entries or just a string
            if type(list(cohort.values())[0]) == list:
                if len(list(cohort.values())[0]) == 1:
                    lst_cohort = list(cohort.values())[0]
                elif len(list(cohort.values())[0]) > 1:
                    lst_cohort = list(cohort.values())[0]
            # When only one column for the cohort should be filtered:
            # Need to provide a list with that one entry only
            elif type(list(cohort.values())[0]) == str:
                lst_cohort = [list(cohort.values())[0]]
            elif type(list(cohort.values())[0]) == dict:
                # This case has to be treated below, since it implies a range.
                # For that, the rows to be removed are filtered by values and not by strings.
                pass
            else:
                raise ValueError(f"Unknown cohort: {cohort}")

            # Filter the metadata and features based on the cohort
            if len(cohort.keys()) == 1 and len(cohort.values()) == 1:
                # If it is a dictionary, the cohort is to be filtered according to a range
                if type(list(cohort.values())[0]) == dict:
                    target_column = list(cohort.keys())[0]
                    dct_range = list(cohort.values())[0]
                    range_keys = list(dct_range.keys())
                    range_values = list(dct_range.values())
                    if (
                        len(range_keys) == 1
                        and len(range_values) == 1
                        and range_keys[0] == "smaller_than"
                    ):
                        # Need to fetch the columns that do NOT contain the desired cohort subpopulation
                        # --> smaller than has to be inverted
                        rows_filter_targets = self.df_files_filtered[
                            self.df_files_filtered[target_column] >= range_values[0]
                        ]
                    elif (
                        len(range_keys) == 1
                        and len(range_values) == 1
                        and range_keys[0] == "greater_than"
                    ):
                        rows_filter_targets = self.df_files_filtered[
                            self.df_files_filtered[target_column] <= range_values[0]
                        ]
                    else:
                        raise ValueError(f"Unknown range: {range_keys}")
                    # Remove the selected rows
                    # Need an empty DataFrame with the scaffold of the features DataFrame to match the structure of the rows_filter_features
                    self._remove_rows_df_idx(
                        self.df_features_filtered.iloc[0:0], rows_filter_targets
                    )
                else:
                    # Need to fetch the columns that do NOT contain the desired cohort subpopulation
                    rows_filter_targets = self.df_files_filtered[
                        ~self.df_files_filtered[list(cohort.keys())[0]].isin(lst_cohort)
                    ]
                    # Empty DataFrame with the scaffold of the features DataFrame to match the structure of the rows_filter_features
                    self._remove_rows_df_idx(
                        self.df_features_filtered.iloc[0:0], rows_filter_targets
                    )
            else:
                raise ValueError(
                    f"Multiple keys and values not yet supported:\n{cohort}"
                )
        else:
            raise ValueError(f"Unknown cohort: {cohort}")
        if list(speech_task.keys())[0] == "all":
            pass
        elif type(speech_task) == dict:
            # Need to fetch the columns that do NOT contain the speech task
            rows_filter_targets = self.df_files_filtered[
                ~self.df_files_filtered["prompt"].isin(list(speech_task.values())[0])
            ]
            # Empty DataFrame with the scaffold of the features DataFrame to match the structure of the rows_filter_features
            self._remove_rows_df_idx(
                self.df_features_filtered.iloc[0:0], rows_filter_targets
            )
        else:
            raise ValueError(f"Unknown speech task: {speech_task}")

    def _remove_rows_df_idx(self, rows_filter_features, rows_filter_targets):
        """
        Remove rows from the indices and label and feature DataFrames based on provided rows that should be removed.
        The rows to be remoed from the feature and label DataFrames to be filtered are combined and removed from the respective DataFrames and indices.

        Args:
            rows_filter_features (pd.DataFrame): The rows to be removed based on the features DataFrame.
            rows_filter_targets (pd.DataFrame): The rows to be removed based on the label DataFrame.
        """
        # Combine the rows that are to be filtered for the features and the target
        rows_filter = rows_filter_features.join(rows_filter_targets, how="outer")

        if not rows_filter.empty:
            print(f"Filtering rows: {len(rows_filter)} rows to be removed")

            # Remove the entries from the respective indices
            train_intersection = self.idx_train_filtered.intersection(rows_filter.index)
            test_intersection = self.idx_test_filtered.intersection(rows_filter.index)
            dev_intersection = self.idx_dev_filtered
            if not self.idx_dev_filtered.empty:
                dev_intersection = self.idx_dev_filtered.intersection(rows_filter.index)
            print(
                f"Length of train intersection: {len(train_intersection)}; length of idx_train: {len(self.idx_train_filtered)}\n"
                f"Length of dev intersection: {len(dev_intersection)}; length of idx_dev: {len(self.idx_dev_filtered)}\n"
                f"Length of test intersection: {len(test_intersection)}; length of idx_test: {len(self.idx_test_filtered)}"
            )

            # Remove the intersecting entries from the respective indices
            self.idx_train_filtered = self.idx_train_filtered.drop(
                train_intersection, errors="ignore"
            )
            self.idx_dev_filtered = self.idx_dev_filtered.drop(
                dev_intersection, errors="ignore"
            )
            self.idx_test_filtered = self.idx_test_filtered.drop(
                test_intersection, errors="ignore"
            )

            # Print the number of removed rows
            print(
                f"Length after removal:\n"
                f"idx_train: {len(self.idx_train_filtered)}\n"
                f"idx_dev: {len(self.idx_dev_filtered)}\n"
                f"idx_test: {len(self.idx_test_filtered)}"
            )
            assert (
                len(test_intersection) + len(train_intersection) + len(dev_intersection)
            ) == len(rows_filter), "Not all rows were removed from the indices"

            print(
                f"Length of df_features before removal: {len(self.df_features_filtered)}\n"
                f"Length of df_files before removal: {len(self.df_files_filtered)}"
            )
            # Remove the respective rows from self.df_features
            self.df_features_filtered = self.df_features_filtered[
                ~self.df_features_filtered.index.isin(rows_filter.index)
            ]

            # Remove the respective rows from self.df_files
            self.df_files_filtered = self.df_files_filtered[
                ~self.df_files_filtered.index.isin(rows_filter.index)
            ]
            print(
                f"Length of df_features after removal:\n {len(self.df_features_filtered)}\n"
                f"Length of df_files after removal: {len(self.df_files_filtered)}\n"
            )
            assert len(self.df_features_filtered) == len(
                self.df_files_filtered
            ), "Length of df_features and df_files do not match"
            assert len(self.idx_train_filtered) + len(self.idx_dev_filtered) + len(
                self.idx_test_filtered
            ) == len(
                self.df_features_filtered
            ), "Length of indices and df_features do not match"
            assert len(self.idx_train_filtered) + len(self.idx_dev_filtered) + len(
                self.idx_test_filtered
            ) == len(
                self.df_files_filtered
            ), "Length of indices and df_files do not match"
