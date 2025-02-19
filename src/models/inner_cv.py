import pandas as pd
import audbenchmark
import numpy as np
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def specificity_score(y_true, y_pred):
    """
    Calculate the specificity score for binary classification.
    In particular: treat the case that all y_true and y_pred only contain one class respectively.
    (this is relevant for LOSO evaluation, where the test partition might be one speaker whose samples only belong to one class)

    The specificity score measures the ability of a model to correctly predict the negative class.
    It is calculated as the ratio of true negatives to the sum of true negatives and false positives.

    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.

    Returns:
    - specificity (float): The specificity score.

    Example:
    >>> y_true = [0, 1, 0, 0, 1]
    >>> y_pred = [0, 1, 1, 0, 0]
    >>> specificity_score(y_true, y_pred)
    0.6666666666666666
    """
    unique_classes_true = np.unique(y_true)
    unique_classes_pred = np.unique(y_pred)

    # Check if both y_true and y_pred contain only one class and it's the same class
    if (
        len(unique_classes_true) == 1
        and len(unique_classes_pred) == 1
        and unique_classes_true[0] == unique_classes_pred[0]
    ):
        # Check if all predictions match the true labels
        # Maybe a bit redundant, but just to be sure
        if np.array_equal(y_true, y_pred):
            # Consider this as perfect prediction
            return 1.0
        else:
            # If there are deviations in predictions
            return 0.0
    else:
        # Calculate specificity as before
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (1, 1):
            # Handle the case where only one class is present differently if needed
            return 0.0
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity


class InnerCV:
    """
    Class representing the inner cross-validation process.

    Parameters:
    - idx_train (pd.MultiIndex): Indices of the training partition.
    - idx_test (pd.MultiIndex): Indices of the test partition.
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target variable.
    - groups (pd.DataFrame): Group labels for group-wise cross-validation.
    - cfg_meta (dict): Meta configuration.
    - target_variable (str): Name of the target variable.
    - cv_strategy_inner (str): Inner cross-validation strategy.
    - cv_inner_method_name (str): Name of the inner cross-validation method (e.g., GridSearchCV).
    - cv_settings_inner (dict): Settings for the inner cross-validation.
    - cv_inner_method_settings (dict): Settings for the inner cross-validation method.
    - estimator_config (dict): Configuration for the estimator.
    - idx_fold (int): Index number of the current outer fold.

    Attributes:
    - idx_train (pd.MultiIndex): Indices of the training partition.
    - idx_test (pd.MultiIndex): Indices of the test partition.
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target variable.
    - groups (pd.DataFrame): Group labels for group-wise cross-validation.
    - cfg_meta (dict): Meta configuration.
    - target_variable (str): Name of the target variable.
    - cv_strategy_inner (str): Inner cross-validation strategy.
    - cv_inner_method_name (str): Name of the inner cross-validation method (e.g., GridSearchCV).
    - cv_settings_inner (dict): Settings for the inner cross-validation.
    - cv_inner_method_settings (dict): Settings for the inner cross-validation method.
    - estimator_config (dict): Configuration for the estimator.
    - idx_fold (int): Index number of the current fold.
    - groups_train (pd.DataFrame): Filtered groups based on the training indices.
    - inner_cv (object): Inner cross-validation splitting object.
    - scorer (object): Scoring method for evaluating the model.
    - estimator (object): Estimator for the inner cross-validation.

    Methods:
    - run_inner_cv(): Perform the inner cross-validation.
    - process_results(): Process the results of the inner cross-validation.
    """

    def __init__(
        self,
        idx_train,
        idx_test,
        X,
        y,
        groups,
        cfg_meta,
        target_variable,
        cv_strategy_inner,
        cv_inner_method_name,
        cv_settings_inner,
        cv_inner_method_settings,
        estimator_config,
        idx_fold,
    ):
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.X = X
        self.y = y
        self.groups = groups

        self.cfg_meta = cfg_meta
        self.target_variable = target_variable
        self.cv_strategy_inner = cv_strategy_inner
        self.cv_inner_method_name = cv_inner_method_name
        self.cv_settings_inner = cv_settings_inner
        self.cv_inner_method_settings = cv_inner_method_settings
        self.estimator_config = estimator_config
        self.idx_fold = idx_fold

        # Filter groups through the indices
        self.groups_train = self.groups.iloc[self.idx_train]

        # Initialise the inner cross validation splitting object
        if self.cv_strategy_inner == "k_fold":
            self.inner_cv = KFold(
                n_splits=self.cv_settings_inner["n_splits"],
                shuffle=self.cv_settings_inner["shuffle"],
                random_state=(
                    self.cv_settings_inner["random_state"]
                    if self.cv_settings_inner["shuffle"]
                    else None
                ),
            )
        elif self.cv_strategy_inner == "group_k_fold":
            self.inner_cv = GroupKFold(
                n_splits=self.cv_settings_inner["n_splits"],
            )
        elif self.cv_strategy_inner == "stratified_group_k_fold":
            self.inner_cv = StratifiedGroupKFold(
                n_splits=self.cv_settings_inner["n_splits"],
                shuffle=self.cv_settings_inner["shuffle"],
                random_state=self.cv_settings_inner["random_state"],
            )
        elif self.cv_strategy_inner == "loso":
            self.inner_cv = LeaveOneGroupOut()
        # None only works when also self.cv_inner_method_name (config: cv_methods_inner) is "no_inner_cv"
        elif self.cv_strategy_inner == "no_inner_cv":
            self.inner_cv = None
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy_inner}")

        # Initialise the scoring method
        if self.estimator_config["problem_type"] == "classification":
            self.scorer = make_scorer(audbenchmark.metric.unweighted_average_recall)
            # self.scorer = make_scorer(specificity_score, greater_is_better=True)

        elif self.estimator_config["problem_type"] == "regression":
            self.scorer = make_scorer(
                audbenchmark.metric.concordance_cc, greater_is_better=True
            )
            # make_scorer(mean_absolute_error, greater_is_better=False)
        else:
            raise ValueError(
                f"Unknown problem type: {self.estimator_config['problem_type']}"
            )

        # Initialise the classifier
        if self.estimator_config["type"] == "KNN":
            self.estimator = KNeighborsClassifier()
        elif self.estimator_config["type"] == "KNeighborsRegressor":
            self.estimator = KNeighborsRegressor()
        elif self.estimator_config["type"] == "SVC":
            self.estimator = SVC(
                class_weight="balanced",
                probability=cfg_meta["predict_proba"],
            )
        elif self.estimator_config["type"] == "SVR":
            self.estimator = SVR()
        elif self.estimator_config["type"] == "LinearRegression":
            self.estimator = LinearRegression()
        elif self.estimator_config["type"] == "RandomForestRegressor":
            self.estimator = RandomForestRegressor()
        elif self.estimator_config["type"] == "RandomForestClassifier":
            self.estimator = RandomForestClassifier()
        elif self.estimator_config["type"] == "DecisionTreeRegressor":
            self.estimator = DecisionTreeRegressor()
        elif self.estimator_config["type"] == "Ridge":
            self.estimator = Ridge()
        elif self.estimator_config["type"] == "XGBClassifier":
            self.estimator = xgb.XGBClassifier()
        elif self.estimator_config["type"] == "XGBRegressor":
            self.estimator = xgb.XGBRegressor()
        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_config['type']}")

    def run_inner_cv(self):
        """
        Perform the inner cross-validation.
        Instantiates self.CV (object): Inner cross-validation object.
        """
        # Perform the inner cross validation
        # Initialise the inner cross validation method
        self.CV = None
        if self.cv_inner_method_name == "GridSearchCV":
            self.CV = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.estimator_config["grid"],
                scoring=self.scorer,
                # TODO: dev would have to be an custom split for inner_cv
                cv=self.inner_cv,
                n_jobs=self.cfg_meta["num_workers"],
                # Check the train score to analyze overfitting
                return_train_score=True,
                verbose=1,
            )
        elif self.cv_inner_method_name == "no_inner_cv":
            # Have to ensure that the config contains only one setting for each parameter
            self.CV = self.estimator.set_params(**self.estimator_config["grid"])
            print(
                f"Using an estimator without inner CV. The following settings have no effect:\n"
                f"self.scorer: {self.scorer}\nself.inner_cv: {self.inner_cv}"
            )
        else:
            raise ValueError(f"Unknown CV method: {self.cv_method_name}")
        # For XGBoostClassifier: need encoded labels
        # if self.estimator_config["type"] == "XGBClassifier":
        #     self.label_encoder = LabelEncoder()
        #     # Have to preserve the index of y for later processing
        #     self.y = pd.Series(
        #         self.label_encoder.fit_transform(self.y), index=self.y.index
        #     )

        if self.estimator_config["type"] == "XGBClassifier":
            self.label_encoder = LabelEncoder()

            # Have to catch the case that some classes are missing in the training partition #
            # Fit the label encoder on the training partition
            self.y_train_encoded = pd.Series(
                self.label_encoder.fit_transform(self.y.iloc[self.idx_train]),
                index=self.y.iloc[self.idx_train].index,
            )

            # Check for missing classes in the training partition
            all_classes = np.unique(self.y)
            train_classes = self.label_encoder.classes_

            # Find missing classes
            missing_classes = np.setdiff1d(all_classes, train_classes)

            # Manually add missing classes to the label encoder
            if missing_classes.size > 0:
                self.label_encoder.classes_ = np.concatenate(
                    [train_classes, missing_classes]
                )

            # Transform the entire dataset using the updated label encoder
            self.y = pd.Series(self.label_encoder.transform(self.y), index=self.y.index)

        # Apply the inner cross validation
        if self.cv_strategy_inner == "no_inner_cv":
            # If have no inner CV: fit the model directly without handing over a groups parameter
            self.CV.fit(
                self.X.iloc[self.idx_train],
                self.y.iloc[self.idx_train],
            )
        else:
            self.CV.fit(
                self.X.iloc[self.idx_train],
                self.y.iloc[self.idx_train],
                groups=self.groups_train[self.cfg_meta["groups"]],
                # TODO sample_weight=sample_weights,
            )

    def process_results(self):
        """
        Process the results of the inner cross-validation.

        Returns:
        - df_results_train (pd.DataFrame): Results of the inner cross-validation on the training data.
        - df_results_test (pd.DataFrame): Results of the inner cross-validation on the test data.
        - CV (object): Inner cross-validation object.
        """
        # Process the results of the inner cross validation
        # Compile all results to DataFrames
        # Get the actual labels
        if self.estimator_config["type"] == "XGBClassifier":
            # Undo label encoding for XGBoostClassifier to be compatible with the evaluation functions
            # Create a Series and then a DataFrame with the reverse-transformed labels
            # --> preserve the original index
            self.df_results_test = pd.DataFrame(
                {
                    self.target_variable: pd.Series(
                        self.label_encoder.inverse_transform(
                            self.y.iloc[self.idx_test]
                        ),
                        index=self.y.iloc[self.idx_test].index,
                    )
                }
            )
            self.df_results_train = pd.DataFrame(
                {
                    self.target_variable: pd.Series(
                        self.label_encoder.inverse_transform(
                            self.y.iloc[self.idx_train]
                        ),
                        index=self.y.iloc[self.idx_train].index,
                    )
                }
            )
        else:
            self.df_results_test = pd.DataFrame(
                {self.target_variable: self.y.iloc[self.idx_test]}
            )
            self.df_results_train = pd.DataFrame(
                {self.target_variable: self.y.iloc[self.idx_train]}
            )

        # Validate best model from train/dev on the test set
        y_pred_test = self.CV.predict(self.X.iloc[self.idx_test])
        # To check for overfitting: also predict the train set
        y_pred_train = self.CV.predict(self.X.iloc[self.idx_train])

        # For XGBoostClassifier: have to undo the label encoding
        # Plotting and presentation of results is more clear with the original labels
        if self.estimator_config["type"] == "XGBClassifier":
            y_pred_test = self.label_encoder.inverse_transform(y_pred_test)
            y_pred_train = self.label_encoder.inverse_transform(y_pred_train)

        # Create a DataFrame with the predictions and the respective indices
        df_y_pred_test = pd.DataFrame({"predictions": y_pred_test}).set_index(
            # Set the index to the original index of the test partition
            # TODO: potentially use range index anyways if have multiple folds
            self.X.iloc[self.idx_test].index
        )
        df_y_pred_train = pd.DataFrame({"predictions": y_pred_train}).set_index(
            self.X.iloc[self.idx_train].index
        )

        df_predict_proba_test = pd.DataFrame()
        df_predict_proba_train = pd.DataFrame()
        if self.cfg_meta["predict_proba"] == True:
            # If desired: predict the probabilities for ROC and PR curves
            predict_proba_test = self.CV.predict_proba(self.X.iloc[self.idx_test])
            df_predict_proba_test = pd.DataFrame(
                # Convert to list to preserve multidimensional array
                {"predict_proba": list(predict_proba_test)}
            ).set_index(self.X.iloc[self.idx_test].index)
            # Train partition
            predict_proba_train = self.CV.predict_proba(self.X.iloc[self.idx_train])
            df_predict_proba_train = pd.DataFrame(
                {"predict_proba": list(predict_proba_train)}
            ).set_index(self.X.iloc[self.idx_train].index)

        # If possible: get the decision function
        df_decision_function_test = pd.DataFrame()
        df_decision_function_train = pd.DataFrame()
        # Classifiers for which no decision function is available
        if not self.estimator_config["type"] in [
            "KNN",
            "KNeighborsRegressor",
            "SVR",
            "LinearRegression",
            "Ridge",
            "random_forrest",
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "XGBRegressor",
        ]:
            if hasattr(self.CV, "decision_function"):
                decision_function = self.CV.decision_function(
                    self.X.iloc[self.idx_test]
                )
                df_decision_function_test = pd.DataFrame(
                    {"decision_function": list(decision_function)}
                ).set_index(self.X.iloc[self.idx_test].index)
                # Train partition
                decision_function_train = self.CV.decision_function(
                    self.X.iloc[self.idx_train]
                )
                df_decision_function_train = pd.DataFrame(
                    {"decision_function": list(decision_function_train)}
                ).set_index(self.X.iloc[self.idx_train].index)
            elif hasattr(self.CV, "predict_proba"):
                print("Decision function not available, using predict_proba instead")
                predict_proba_test = self.CV.predict_proba(self.X.iloc[self.idx_test])
                df_decision_function_test = pd.DataFrame(
                    predict_proba_test,
                    columns=[
                        f"prob_class_{i}" for i in range(predict_proba_test.shape[1])
                    ],
                ).set_index(self.X.iloc[self.idx_test].index)
                # Train partition
                predict_proba_train = self.CV.predict_proba(self.X.iloc[self.idx_train])
                df_decision_function_train = pd.DataFrame(
                    predict_proba_train,
                    columns=[
                        f"prob_class_{i}" for i in range(predict_proba_train.shape[1])
                    ],
                ).set_index(self.X.iloc[self.idx_train].index)

        # Concatenate all relevant result DataFrames
        self.df_results_test = pd.concat(
            [
                self.df_results_test,
                df_y_pred_test,
                df_predict_proba_test,
                df_decision_function_test,
            ],
            axis=1,
        )
        self.df_results_train = pd.concat(
            [
                self.df_results_train,
                df_y_pred_train,
                df_predict_proba_train,
                df_decision_function_train,
            ],
            axis=1,
        )

        # Add the fold index for tracking it in the outer CV
        self.df_results_test["fold_index"] = self.idx_fold
        self.df_results_train["fold_index"] = self.idx_fold

        # Add the grouping labels (e.g., speaker) for later group-wise voting
        self.df_results_test[self.cfg_meta["groups"]] = self.groups[
            self.cfg_meta["groups"]
        ].iloc[self.idx_test]
        self.df_results_train[self.cfg_meta["groups"]] = self.groups[
            self.cfg_meta["groups"]
        ].iloc[self.idx_train]
        # If a session column for further grouping is available: add it to the results
        if self.cfg_meta["sessions"] is not None:
            self.df_results_test[self.cfg_meta["sessions"]] = self.groups[
                self.cfg_meta["sessions"]
            ].iloc[self.idx_test]
            self.df_results_train[self.cfg_meta["sessions"]] = self.groups[
                self.cfg_meta["sessions"]
            ].iloc[self.idx_train]

        return self.df_results_train, self.df_results_test, self.CV
