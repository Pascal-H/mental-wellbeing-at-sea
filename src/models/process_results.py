import audbenchmark
import audeer

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy import stats
import seaborn as sns
import shutil
import yaml

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from models.confusion_matrix import make_confusion_matrix


class Results:
    """
    Class to process and store the results of a modeling experiment.

    Args:
        cfg_meta (dict): The metadata configuration.
        df_features_filtered (pd.DataFrame): The filtered features DataFrame.
        df_files_filtered (pd.DataFrame): The filtered files DataFrame with labels.
        groups (pd.DataFrame): The groups (e.g., speaker) used for modeling.
        target_variable (str): The target variable.
        df_results_outer_test (pd.DataFrame): The results DataFrame for the test set of the outer CV.
        df_results_outer_train (pd.DataFrame): The results DataFrame for the train set of the outer CV.
        df_results_maxvote (pd.DataFrame): The results DataFrame for grouping the predictions by max vote.
        dct_inner_cv_objects (dict): A dictionary, containing the inner CV object for each inner fold index number.
        estimator_config (dict): The configuration for the estimator.

    Raises:
        ValueError: If the `problem_type` in `estimator_config` is unknown.

    Attributes:
        cfg_meta (dict): The metadata configuration.
        df_features_filtered (pd.DataFrame): The filtered features DataFrame.
        df_files_filtered (pd.DataFrame): The filtered files DataFrame with labels.
        groups (pd.DataFrame): The groups (e.g., speaker) used for modeling.
        target_variable (str): The target variable.
        df_results_outer_test (pd.DataFrame): The results DataFrame for the test set of the outer CV.
        df_results_outer_train (pd.DataFrame): The results DataFrame for the train set of the outer CV.
        df_results_maxvote (pd.DataFrame): The results DataFrame for grouping the predictions by max vote.
        dct_inner_cv_objects (dict): A dictionary, containing the inner CV object for each inner fold index number.
        estimator_config (dict): The configuration for the estimator.
        metrics (dict): The dictionary of evaluation metrics based on the problem type.

    Methods:
        compile_results: Compiles the results by saving the Python-based classification objects, calculating performance metrics, and plotting.
        _save_python_objects: Saves Python objects and DataFrames.
        _calculate_performance_metrics: Calculates various performance metrics based on the model predictions.
        _calculate_binary_classification_metrics: Calculates detailed classification metrics only available for binary classification problems.
        _plot_confusion_matrices: Plots confusion matrices based on the predictions.
        _plot_roc: Plots ROC curves based on the predictions.
    """

    def __init__(
        self,
        cfg_meta,
        df_features_filtered,
        df_files_filtered,
        groups,
        target_variable,
        df_results_outer_test,
        df_results_outer_train,
        df_results_maxvote,
        dct_inner_cv_objects,
        estimator_config,
    ):
        self.cfg_meta = cfg_meta
        self.df_features_filtered = df_features_filtered
        self.df_files_filtered = df_files_filtered
        self.groups = groups
        self.target_variable = target_variable
        self.df_results_test = df_results_outer_test
        self.df_results_train = df_results_outer_train
        self.df_results_maxvote = df_results_maxvote
        self.dct_inner_cv_objects = dct_inner_cv_objects
        self.estimator_config = estimator_config

        # Initialise the evaluation metrics depending on the problem type
        if self.estimator_config["problem_type"] == "classification":
            self.metrics = {
                "unweighted_average_recall": audbenchmark.metric.unweighted_average_recall,
                "accuracy": audbenchmark.metric.accuracy,
                "unweighted_average_fscore": audbenchmark.metric.unweighted_average_fscore,
                "roc_auc_score": roc_auc_score,
                "average_precision_score": average_precision_score,
            }
        elif self.estimator_config["problem_type"] == "regression":
            self.metrics = {
                "pearson_cc": audbenchmark.metric.pearson_cc,
                "concordance_cc": audbenchmark.metric.concordance_cc,
                "mean_squared_error": audbenchmark.metric.mean_squared_error,
                "mean_absolute_error": audbenchmark.metric.mean_absolute_error,
            }
        else:
            raise ValueError(
                f"Unknown problem type: {self.estimator_config['problem_type']}"
            )

    def compile_results(
        self, path_results, path_config_experiment, lst_configs_run, config_modelling
    ):
        """
        Compiles the results by saving classification-related Python objects, calculating performance metrics, and plotting.

        Args:
            path_results (str): The path to save the results at.
            path_config_experiment (str): The path to the experiment configuration file.
            lst_configs_run (list): The list of the current whole experiment meta configuration for composing the results output path.
            config_modelling (dict): The configuration for the current modelling run.
        """
        self._save_python_objects(
            path_results, path_config_experiment, lst_configs_run, config_modelling
        )

        # Calculate the performance metrics and plot the results over all possible folds
        self._calculate_performance_metrics()
        self._plot_confusion_matrices()
        self._plot_roc()
        self._plot_pr()
        self._plot_regplot()

        ## Fold-specific ##
        # If several folds exist: calculate model performance per fold and avg and sd, and box plots of these individual models
        # Back up the main DataFrames
        self.df_results_test_full = self.df_results_test.copy()
        self.path_plots_full = self.path_plots
        self.path_models_full = self.path_models
        self.df_results_train_full = self.df_results_train
        self.df_results_maxvote_full = self.df_results_maxvote

        lst_results_all = []
        if len(self.df_results_test_full.fold_index.unique()) != 1:
            # Iterate over each fold and calculate the performance metrics and plots for each fold
            for self.cur_fold in list(self.df_results_test_full.fold_index.unique()):
                # Re-assign the DataFrames and paths for each fold
                self.path_models = audeer.mkdir(
                    os.path.join(
                        self.path_models_full, "folds", f"fold-{self.cur_fold}"
                    )
                )
                self.path_plots = audeer.mkdir(
                    os.path.join(self.path_plots_full, "folds", f"fold-{self.cur_fold}")
                )
                # Re-assign the DataFrames for each fold
                self.df_results_train = self.df_results_train_full[
                    self.df_results_train_full.fold_index == self.cur_fold
                ]
                # self.df_results_maxvote = self.df_results_maxvote_full[
                #     self.df_results_maxvote_full.fold_index == self.cur_fold
                # ]
                self.df_results_test = self.df_results_test_full[
                    self.df_results_test_full.fold_index == self.cur_fold
                ]
                self._calculate_performance_metrics()
                # Concatenate all fold-specific performance metrics result dicts
                lst_results_all.append(self.dct_results)
                self._plot_confusion_matrices()
                self._plot_roc()
                self._plot_pr()
                self._plot_regplot()

            # Aggregate all results over the folds
            compiled_results = self._compile_results_dict(lst_results_all)
            with open(
                os.path.join(self.path_models_full, "results-compiled.yaml"), "w"
            ) as file:
                yaml.dump(compiled_results, file)

            # Aggregate the model performance to a boxplot
            self._plot_boxplot_folds(compiled_results)

        else:
            # Create a file "results-compiled" with average and std nevertheless to enable late aggregation of all results
            with open(
                os.path.join(self.path_models_full, "results-compiled.yaml"), "w"
            ) as file:
                yaml.dump(self.dct_results, file)

    def _compile_results_dict(self, lst_results_all):
        """
        Compiles and aggregates results from a list of dictionaries into a single dictionary with statistical analysis.

        This method takes a list of dictionaries, where each dictionary represents a set of results (e.g., from different
        test partitions or iterations of a process). It compiles these results into a single dictionary that aggregates
        the raw values and computes the average and standard deviation for each key across all dictionaries. This method
        is particularly useful for summarizing results from experiments that involve multiple runs, such as cross-validation
        or bootstrapping.

        Parameters:
            lst_results_all (list of dict): A list where each element is a dictionary containing the results of a single
                experiment run. Each dictionary should have the same structure, where the keys
                are the metric names and the values are the metric values.

        Returns:
            dict: A dictionary where each key is a metric name from the input dictionaries. Each key maps to a dictionary
                that contains a list of raw values (under the key "raw_values") from all input dictionaries for that metric,
                as well as the calculated average (under the key "average") and standard deviation (under the key "sd")
                of those values. If the raw values for a metric contain only `None`, both the average and standard deviation
                are set to `None` to indicate missing or non-applicable data.

        Note:
            - The method filters out `None` values before calculating the average and standard deviation, ensuring that
            these statistics are only computed from valid numeric values.
            - If all values for a given key are `None`, the method sets the average and standard deviation for that key
            to `None`, indicating that no valid data was available for those calculations.
        """
        compiled_results = {}

        for dct in lst_results_all:
            for key, value in dct.items():
                if key not in compiled_results:
                    compiled_results[key] = {"raw_values": []}
                compiled_results[key]["raw_values"].append(value)

        for key, values_dict in compiled_results.items():
            # Filter out None values
            raw_values = [
                value for value in values_dict["raw_values"] if value is not None
            ]

            # Check if raw_values is empty after filtering
            if raw_values:
                avg = float(np.mean(raw_values))
                sd = float(np.std(raw_values))
            else:
                # Handle the case where raw_values is empty, e.g., by setting avg and sd to None
                avg = None
                sd = None

            values_dict.update({"average": avg, "sd": sd})

        return compiled_results

    def _save_python_objects(
        self, path_results, path_config_experiment, lst_configs_run, config_modelling
    ):
        """
        Save various Python objects related to the experiment.
        Add the current modelling config to the list of configuration parameters of this very current experiment run and save it as an YAML for later readability.
        Save the data used in this experiment as parquet files for later reproducibility.
        Save the results DataFrames as parquet files for later reproducibility.
        Save the dictionary with the inner CV objects to pickle for later reproducibility.

        Args:
            path_results (str): The path to the results directory.
            path_config_experiment (str): The path to the config file for the overall experiment.
            lst_configs_run (list): A list of dictionaries containing the current experiment meta parameters.
            config_modelling (dict): A tuple containing the model configuration parameters.
        """
        path_data = audeer.mkdir(os.path.join(path_results, "data"))
        self.path_models = audeer.mkdir(os.path.join(path_results, "models"))
        self.path_plots = audeer.mkdir(os.path.join(path_results, "plots"))

        # Copy the entire config file for the overall experiment
        shutil.copy2(path_config_experiment, self.path_models)

        # Compose and save a dictionary with the current experiment meta parameters and model config
        # Map the parameters tuple back to the parameter name for the modelling config
        # Import here to avoid circular import error
        from utils import map_modelling_config

        lst_configs_run.append(map_modelling_config(config_modelling))
        dct_config_experiment = {}
        for config in lst_configs_run:
            dct_config_experiment.update(config)

        # Save the dictionary with the current meta parameters and model config to YAML
        with open(
            os.path.join(self.path_models, "experiment_config.yaml"), "w"
        ) as file:
            yaml.dump(dct_config_experiment, file)

        # Save the DataFrames with the data basis in a compressed way for later reproducibility
        if self.cfg_meta["save_full_data"] == True:
            self.df_features_filtered.to_parquet(
                os.path.join(path_data, "df_features_filtered.parquet.zstd"),
                compression="zstd",
            )
            self.df_files_filtered.to_parquet(
                os.path.join(path_data, "df_files_filtered.parquet.zstd"),
                compression="zstd",
            )

        # Hot fix: sometimes only have a float instead of an ndarray in decision_function
        # Define a custom function to convert float to np.ndarray
        def ensure_array(value):
            if isinstance(value, float):
                return np.array([value])
            return value

        # Save the DataFrames with the results
        if "decision_function" in self.df_results_test.columns:
            self.df_results_test["decision_function"] = self.df_results_test[
                "decision_function"
            ].apply(ensure_array)
        self.df_results_test.to_parquet(
            os.path.join(path_data, "df_results_test.parquet.zstd"),
            compression="zstd",
        )
        if "decision_function" in self.df_results_train.columns:
            self.df_results_train["decision_function"] = self.df_results_train[
                "decision_function"
            ].apply(ensure_array)
        self.df_results_train.to_parquet(
            os.path.join(path_data, "df_results_train.parquet.zstd"),
            compression="zstd",
        )
        self.df_results_maxvote.to_parquet(
            os.path.join(path_data, "df_results_maxvote.parquet.zstd"),
            compression="zstd",
        )

        # Save the dictionary inner CV objects to pickle
        with open(
            os.path.join(self.path_models, "dct_inner_cv_objects.pkl"), "wb"
        ) as file:
            pickle.dump(self.dct_inner_cv_objects, file)

    def _calculate_performance_metrics(self):
        """
        Calculates performance metrics for the model based on the provided model predictions.

        This method calculates various performance metrics for the model, including ROC AUC score,
        average precision score, and regular metrics. It iterates over different sets (test, train,
        max_vote) and calculates the metrics for each set. It also calculates detailed binary
        classification metrics if the target variable has two unique values.
        """
        # Dictionary to store summary results
        dct_results = {}

        # Calculate common scores across all classified samples
        # Iterate over all sets
        for cur_set in ["test", "train", "max_vote"]:
            if cur_set == "test":
                df_results = self.df_results_test
            elif cur_set == "train":
                df_results = self.df_results_train
            elif cur_set == "max_vote":
                df_results = self.df_results_maxvote

            for cur_metric in self.metrics.keys():
                # Only do ROC and precision scores for binary classification
                # max_vote does not have prediction probabilities
                if cur_metric == "roc_auc_score":
                    # TODO: label encoder
                    if cur_set == "max_vote":
                        continue
                    if len(self.df_files_filtered[self.target_variable].unique()) == 2:
                        dct_results[f"{cur_metric}-{cur_set}"] = (
                            self.calculate_roc_auc_score(
                                df_results, self.target_variable
                            )
                        )
                elif cur_metric == "average_precision_score":
                    if cur_set == "max_vote":
                        continue
                    if len(self.df_files_filtered[self.target_variable].unique()) == 2:
                        dct_results[f"{cur_metric}-{cur_set}"] = (
                            self.calculate_average_precision_score(
                                df_results, self.target_variable
                            )
                        )
                else:
                    dct_results[f"{cur_metric}-{cur_set}"] = (
                        self.calculate_regular_metric(
                            df_results, self.target_variable, cur_metric
                        )
                    )

            # Calculate detailed binary classification metrics
            if len(self.df_files_filtered[self.target_variable].unique()) == 2:
                dct_results = self._calculate_binary_classification_metrics(
                    df_results, cur_set, dct_results
                )

        # Instantiate the results dictionary to use it for later ROC and PR curve labelling
        self.dct_results = dct_results

        # Save the results dictionary as a YAML
        with open(os.path.join(self.path_models, "results.yaml"), "w") as file:
            yaml.dump(dct_results, file)

    def _calculate_binary_classification_metrics(
        self, df_results, set_specifier, dct_results
    ):
        """
        Calculates and returns binary classification metrics based on the provided predictions and ground truth.

        This function is designed to handle binary classification scenarios, including those with imbalanced or
        single-class data distributions. It is particularly useful in cases such as Leave-One-Speaker-Out (LOSO)
        validation, where the test partition may contain instances from only one class. The function ensures accurate
        performance evaluation by appropriately handling these cases.

        Parameters:
            predictions (list of int): A list of predicted labels by the model, where each label is either 0 or 1.
            ground_truth (list of int): A list of actual labels, where each label is either 0 or 1.

        Returns:
            dict: A dictionary containing the calculated metrics, including accuracy, precision, recall, and F1 score.
                Each metric is keyed by its name as a string.

        Note:
            The function assumes that the input lists, predictions and ground_truth, are of equal length and contain
            only binary values. It does not perform input validation, and it's the caller's responsibility
            to ensure the input data's integrity.
        """
        # Calculate confusion matrix
        conf_mat = confusion_matrix(
            df_results[self.target_variable], df_results["predictions"]
        )

        # Initialize TN, FN, TP, FP
        TN, FN, TP, FP = 0, 0, 0, 0

        # Check the shape of the confusion matrix to handle different scenarios
        if conf_mat.shape == (2, 2):
            # Standard case: binary classification with both classes present
            TN, FN, TP, FP = (
                conf_mat[0][0],
                conf_mat[1][0],
                conf_mat[1][1],
                conf_mat[0][1],
            )
        elif conf_mat.shape == (1, 1):
            # Determine the actual and predicted classes present
            actual_class_present = df_results[self.target_variable].unique()[0]
            predicted_class_present = df_results["predictions"].unique()[0]

            # Correctly assign TN or TP based on the actual and predicted classes
            if actual_class_present == 0 and predicted_class_present == 0:
                TN = conf_mat[0][0]
            elif actual_class_present == 1 and predicted_class_present == 1:
                TP = conf_mat[0][0]
            # Handle cases where predictions are all one class but actuals are the opposite
            elif actual_class_present == 0 and predicted_class_present == 1:
                FP = conf_mat[0][0]
            elif actual_class_present == 1 and predicted_class_present == 0:
                FN = conf_mat[0][0]

        # Helper function to safely calculate division and handle division by zero
        def safe_divide(numerator, denominator):
            return float(numerator) / float(denominator) if denominator else None

        # Calculate metrics safely
        metrics = {
            f"Sensitivity/TPR-{set_specifier}": safe_divide(TP, TP + FN),
            f"Specificity/TNR-{set_specifier}": safe_divide(TN, TN + FP),
            f"Precision/PPV-{set_specifier}": safe_divide(TP, TP + FP),
            f"NPV-{set_specifier}": safe_divide(TN, TN + FN),
            f"FPR-{set_specifier}": safe_divide(FP, FP + TN),
            f"FNR-{set_specifier}": safe_divide(FN, TP + FN),
            f"FDR-{set_specifier}": safe_divide(FP, TP + FP),
        }
        dct_results.update(metrics)

        return dct_results

    def _plot_confusion_matrices(self):
        """
        Plot confusion matrices for the test and train partitions and for the max voting results.

        This method calculates the confusion matrix using SKLearn and plots it using a convenience plotting function.
        The resulting confusion matrices are saved in different formats.
        """
        if self.estimator_config["problem_type"] == "classification":
            # Plot confusion matrices for the test and train set and for the max voting results
            for cur_set in ["test", "train", "max_vote"]:
                if cur_set == "test":
                    df_results = self.df_results_test
                elif cur_set == "train":
                    df_results = self.df_results_train
                elif cur_set == "max_vote":
                    df_results = self.df_results_maxvote

                # Calculate the confusion matrix with SKLearn
                conf_mat = confusion_matrix(
                    # TODO: label encoder
                    df_results[self.target_variable],
                    df_results["predictions"],
                    labels=df_results[self.target_variable].unique(),
                )
                # Plot a confusion matrix
                # Call a convenient plotting function from
                # https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
                cur_fig, cur_ax = make_confusion_matrix(
                    conf_mat,
                    title=f"Confusion Matrix",
                    categories=df_results[self.target_variable].unique(),
                    figsize=(4, 4),
                    cbar=False,
                    sum_stats=False,
                )
                # Save the figure
                for cur_extension in [".png", ".pdf", ".svg"]:
                    cur_fig.savefig(
                        os.path.join(
                            self.path_plots, f"confmat-{cur_set}{cur_extension}"
                        ),
                        bbox_inches="tight",
                    )
                plt.close(cur_fig)

    def _plot_roc(self):
        """
        Plot the Receiver Operating Characteristic (ROC) curves for the classification model.

        This method plots the ROC curve for the test and train partitions of the classification model.
        It calculates the False Positive Rate (FPR) and True Positive Rate (TPR) for each partition,
        and plots the curves on the same graph. It also includes the Area Under the Curve (AUC) score
        for each partition in the legend.
        The labels are scaled according to the figure size, and the figure is saved in different formats.
        """
        if (
            self.estimator_config["problem_type"] == "classification"
            and len(self.df_files_filtered[self.target_variable].unique()) == 2
        ):
            # Initialise the plotting object
            roc_fig, roc_ax = plt.subplots(figsize=(4, 4))
            # Get the figure size to scale the fonts later
            fig_width, fig_height = roc_fig.get_size_inches()
            # Initialise the line colors for the different partitions
            dct_colours = {
                "test": "green",
                "train": "blue",
            }

            # Iterate over the different sets and plot their curves on top of each other
            # If only a subset combination is to be plotted: implement another loop on subsets
            # "max_vote" not possible, since it does not have prediction probabilities
            # TODO: handle classifiers that don't have predict_proba (e.g. KNN?)
            for cur_set in ["test", "train"]:
                if cur_set == "test":
                    df_results = self.df_results_test
                elif cur_set == "train":
                    df_results = self.df_results_train

                roc_fpr, roc_tpr, roc_thresholds = self.calculate_roc_curve(
                    df_results, self.target_variable
                )

                # Check if the ROC AUC score is None and handle it appropriately
                roc_auc_score = self.dct_results.get(f"roc_auc_score-{cur_set}")
                if roc_auc_score is not None:
                    roc_auc_score_str = f"{round(roc_auc_score, 3)}"
                else:
                    roc_auc_score_str = "N/A"

                roc_ax.plot(
                    roc_fpr,
                    roc_tpr,
                    lw=min(fig_width, fig_height) * 0.3,
                    alpha=0.5,
                    color=dct_colours[cur_set],
                    label=(
                        f"ROC curve - {cur_set}; " f"AUC score: " f"{roc_auc_score_str}"
                    ),
                )

            roc_ax.plot(
                [0, 1],
                [0, 1],
                alpha=0.25,
                linestyle="--",
                color="red",
                lw=min(fig_width, fig_height) * 0.3,
                label="Chance level",
            )
            roc_ax.set_xlim(xmin=-0.01, xmax=1.01)
            roc_ax.set_ylim(ymin=-0.01, ymax=1.01)
            roc_ax.set_ylabel(
                "True positive rate",
                fontsize=min(fig_width, fig_height) * 2.5,
            )
            roc_ax.set_xlabel(
                "False positive rate",
                fontsize=min(fig_width, fig_height) * 2.5,
            )
            roc_ax.tick_params(
                axis="both",
                labelsize=min(fig_width, fig_height) * 2,
            )
            roc_ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fontsize=min(fig_width, fig_height) * 2,
            )
            roc_ax.set_title(
                f"ROC curve for all samples",
                fontsize=min(fig_width, fig_height) * 3,
            )

            for cur_extension in [".png", ".pdf", ".svg"]:
                roc_fig.savefig(
                    os.path.join(self.path_plots, f"roc_curve{cur_extension}"),
                    bbox_inches="tight",
                )

            plt.close(roc_fig)

    def _plot_pr(self):
        """
        Plot the Precision-Recall curves for the test and train partitions.

        This function calculates the Precision-Recall curves for the test and train partitions
        and plots them on a single figure. It also includes the average precision score
        for each set and the chance level line.
        The labels are scaled according to the figure size, and the figure is saved in different formats.
        """
        if self.df_results_train[self.target_variable].unique().size != 2:
            return
        pr_fig, pr_ax = plt.subplots(figsize=(4, 4))
        # Get the figure size to scale the fonts later
        fig_width, fig_height = pr_fig.get_size_inches()
        # Initialise the line colors for the different partitions
        dct_colours = {
            "test": "green",
            "train": "blue",
        }

        for cur_set in ["test", "train"]:
            if cur_set == "test":
                df_results = self.df_results_test
            elif cur_set == "train":
                df_results = self.df_results_train

            pr_precision, pr_recall, pr_thresholds = (
                self.calculate_precision_recall_curve(df_results, self.target_variable)
            )
            average_precision_score = self.calculate_average_precision_score(
                df_results, self.target_variable
            )

            # Check if the average precision score is None and handle it appropriately
            if average_precision_score is not None:
                average_precision_score_str = f"{round(average_precision_score, 3)}"
            else:
                average_precision_score_str = "N/A"

            pr_ax.plot(
                pr_recall,
                pr_precision,
                lw=min(fig_width, fig_height) * 0.3,
                alpha=0.5,
                color=dct_colours[cur_set],
                label=(
                    f"Precision-Recall curve - {cur_set};\n"
                    f"avg. precision score: "
                    f"{average_precision_score_str}"
                ),
            )

            # Calculate the mode of the target variable in the test set
            majority_class = stats.mode(
                df_results[self.target_variable], keepdims=False
            )[0]
            # Find the unique labels in the target variable
            unique_labels = df_results[self.target_variable].unique()
            # Attempt to identify the positive class as the minority class
            positive_class_candidates = [
                label for label in unique_labels if label != majority_class
            ]

            if positive_class_candidates:
                positive_class = positive_class_candidates[0]
                # Calculate the no skill line as the proportion of the positive class in the test set
                no_skill = sum(
                    df_results[self.target_variable] == positive_class
                ) / len(df_results[self.target_variable])
                # Plot the no skill line
                pr_ax.plot(
                    [0, 1],
                    [no_skill, no_skill],
                    lw=min(fig_width, fig_height) * 0.3,
                    alpha=0.25,
                    linestyle="--",
                    color=dct_colours[cur_set],
                    label=f"Chance level - {cur_set}",
                )
            else:
                print(
                    f"Warning: Unable to identify a positive class for {cur_set} set. Chance level line will not be plotted."
                )

        pr_ax.set_xlim(xmin=-0.01, xmax=1.01)
        pr_ax.set_ylim(ymin=-0.01, ymax=1.01)
        pr_ax.set_ylabel("Precision", fontsize=min(fig_width, fig_height) * 2.5)
        pr_ax.set_xlabel("Recall", fontsize=min(fig_width, fig_height) * 2.5)
        pr_ax.tick_params(axis="both", labelsize=min(fig_width, fig_height) * 2)
        pr_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=min(fig_width, fig_height) * 2,
        )
        pr_ax.set_title(
            "Precision-Recall curve for all samples",
            fontsize=min(fig_width, fig_height) * 3,
        )

        for cur_extension in [".png", ".pdf", ".svg"]:
            pr_fig.savefig(
                os.path.join(self.path_plots, f"pr_curve{cur_extension}"),
                bbox_inches="tight",
            )

        plt.close(pr_fig)

    def _plot_regplot(self):
        """
        Plot regression plots for the test and train set and for the max voting results.

        This method generates regression plots for the test set, train set, and max voting results.
        The plots show the relationship between the predicted values and the target variable.
        """
        if self.estimator_config["problem_type"] == "regression":
            # Plot regression plots for the test and train set and for the max voting results
            for cur_set in ["test", "train", "max_vote"]:
                if cur_set == "test":
                    df_results = self.df_results_test
                elif cur_set == "train":
                    df_results = self.df_results_train
                # elif cur_set == "max_vote":
                #     df_results = self.df_results_maxvote

                reg_fig, reg_ax = plt.subplots(figsize=(4, 4))
                # Get the figure size to scale the fonts later
                fig_width, fig_height = reg_fig.get_size_inches()

                sns.regplot(
                    x="predictions", y=self.target_variable, data=df_results, ax=reg_ax
                )

                reg_ax.set_ylabel(
                    "Target",
                    fontsize=min(fig_width, fig_height) * 2.5,
                )
                reg_ax.set_xlabel(
                    "Prediction",
                    fontsize=min(fig_width, fig_height) * 2.5,
                )
                reg_ax.tick_params(
                    axis="both",
                    labelsize=min(fig_width, fig_height) * 2,
                )
                reg_ax.set_title(
                    f"Regression Plot",
                    fontsize=min(fig_width, fig_height) * 3,
                )

                # Save the figure
                for cur_extension in [".png", ".pdf", ".svg"]:
                    reg_fig.savefig(
                        os.path.join(
                            self.path_plots, f"regplot-{cur_set}{cur_extension}"
                        ),
                        bbox_inches="tight",
                    )
                plt.close(reg_fig)

        return

    def _plot_boxplot_folds(self, compiled_results):
        path_boxplots = audeer.mkdir(
            os.path.join(self.path_plots_full, "boxplots-folds")
        )
        for metric in self.metrics.keys():
            metric_results = {k: v for k, v in compiled_results.items() if metric in k}

            # Initialize a list to store filtered values
            filtered_metric_results = {}

            # Filter out None values or empty lists from metric_results
            for result, values_dict in metric_results.items():
                if "max_vote" in result:
                    continue
                filtered_values = [
                    v for v in values_dict["raw_values"] if v is not None
                ]
                if filtered_values:  # Ensure there are non-None values
                    filtered_metric_results[result] = {"raw_values": filtered_values}

            # Proceed only if there are metrics to plot
            if not filtered_metric_results:
                print(f"No data to plot for metric {metric}. Skipping.")
                continue

            boxplot_fig, boxplot_ax = plt.subplots(figsize=(4, 4))
            fig_width, fig_height = boxplot_fig.get_size_inches()

            for i, (result, values_dict) in enumerate(filtered_metric_results.items()):
                boxplot_ax.boxplot(
                    values_dict["raw_values"],
                    positions=[i],
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightgrey"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    medianprops=dict(color="black"),
                )

            boxplot_ax.set_xticks(range(len(filtered_metric_results)))
            boxplot_ax.set_xticklabels(filtered_metric_results.keys(), rotation=45)
            boxplot_ax.set_ylabel(
                "Metric value", fontsize=min(fig_width, fig_height) * 2.5
            )
            boxplot_ax.set_xlabel("Metric", fontsize=min(fig_width, fig_height) * 2.5)
            boxplot_ax.tick_params(
                axis="both", labelsize=min(fig_width, fig_height) * 2
            )
            boxplot_ax.set_title(
                f"Performance metrics over the different folds for {metric}",
                fontsize=min(fig_width, fig_height) * 3,
            )

            for cur_extension in [".png", ".pdf", ".svg"]:
                boxplot_fig.savefig(
                    os.path.join(
                        path_boxplots, f"boxplot-folds-{metric}{cur_extension}"
                    ),
                    bbox_inches="tight",
                )
            plt.close(boxplot_fig)

    def calculate_roc_auc_score(self, df_results, target):
        """
        Calculate the ROC AUC score for the given dataframe and target column.

        Parameters:
            df_results (pd.DataFrame): The dataframe containing the results.
            target (str): The name of the column holding target variable.

        Returns:
            float: The calculated ROC AUC score, or None if only one class is present.
        """
        try:
            return float(
                self.metrics["roc_auc_score"](
                    df_results[target],
                    df_results["predict_proba"].apply(lambda row: row[1]),
                )
            )
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                # Handle the case where only one class is present
                # Option 1: Return None
                return None
                # Option 2: Log the error and return None or a default value
                # print(f"Cannot calculate ROC AUC score: {e}")
                # return None
            else:
                # Re-raise the exception if it's not the expected error
                raise

    def calculate_average_precision_score(self, df_results, target):
        """
        Calculate the average precision score for a given target variable.

        Args:
            df_results (pandas.DataFrame): The DataFrame containing the results.
            target (str): The name of the column holding target variable.

        Returns:
            float or None: The average precision score, or None if an error occurs.
        """
        try:
            positive_class = df_results[target].mode()[0]
            average_precision = float(
                self.metrics["average_precision_score"](
                    df_results[target],
                    df_results["predict_proba"].apply(lambda row: row[1]),
                    pos_label=positive_class,
                )
            )
            return average_precision
        except Exception as e:
            # Log the error and return None
            print(f"Cannot calculate average precision score: {e}")
            return None

    def calculate_roc_curve(self, df_results, target):
        """
        Calculate the Receiver Operating Characteristic (ROC) curve.

        Parameters:
            df_results (pandas.DataFrame): The DataFrame containing the results.
            target (str): The name of the column holding target variable.

        Returns:
            tuple: A tuple containing three arrays: fpr (False Positive Rate), tpr (True Positive Rate),
                   and thresholds (threshold values).
        """
        positive_class = df_results[target].mode()[0]
        return roc_curve(
            df_results[target],
            df_results["predict_proba"].apply(lambda row: row[1]),
            pos_label=positive_class,
        )

    def calculate_precision_recall_curve(self, df_results, target):
        """
        Calculate the precision-recall curve for a given target variable.

        Args:
            df_results (pandas.DataFrame): The DataFrame containing the results.
            target (str): The name of the column holding target variable.

        Returns:
            tuple: A tuple containing three arrays: precision, recall, and thresholds.
        """
        positive_class = df_results[target].mode()[0]
        return precision_recall_curve(
            df_results[target],
            df_results["predict_proba"].apply(lambda row: row[1]),
            pos_label=positive_class,
        )

    def calculate_regular_metric(self, df_results, target, metric):
        """
        Calculate a regular metric for evaluating model performance.

        Args:
            df_results (pandas.DataFrame): The DataFrame containing the results.
            target (str): The name of the column holding target variable.
            metric (object): The metric object to be used to calculate the classification performance.

        Returns:
            The calculated metric value.
        """
        return self.metrics[metric](df_results[target], df_results["predictions"])
