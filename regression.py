import sklearn
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from IPython.utils import io

from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.feature_selection import (
    RFE,
    RFECV,
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
    f_regression,
    SelectFromModel,
    f_classif,
    chi2,
)

from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
    cross_val_score,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    StackingRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBRegressor

import copy


class TaskSolver:
    def __init__(
        self,
        train_df,
        test_df,
        imputer,
        outliers_detector,
        scaler,
        corr_threshold,
        feature_selector,
        verbose=False,  # Enable all message printing
    ):
        self.train_df = train_df
        self.train_features = train_df.drop(
            ["id", "y", "class", "fold"], axis=1
        ).to_numpy()
        self.train_targets = train_df["y"]
        self.train_folds = train_df["fold"]
        self.test_df = test_df
        self.test_features = test_df.drop(["id"], axis=1).to_numpy()
        self.imputer = imputer
        self.outliers_detector = outliers_detector
        self.scaler = scaler
        self.corr_threshold = corr_threshold
        self.feature_selector = feature_selector
        self.verbose = verbose

    def _print(self, *args):
        if self.verbose:
            print(*args)
        else:
            pass

    def _count_missing_values(self, x):
        missing_count = np.count_nonzero(np.isnan(x))
        total_entries = x.shape[0] * x.shape[1]
        missing_pct = missing_count / total_entries
        self._print(
            f"Total missing values: {missing_count}/{total_entries} ({missing_pct*100:.3f}%)"
        )
        return missing_pct

    def _corr2_coeff(self, A, B):
        # credit: https://stackoverflow.com/a/30143754/7053239
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]
        # Sum of squares across rows
        ssA = (A_mA ** 2).sum(1)
        ssB = (B_mB ** 2).sum(1)
        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    def _preprocess_data(self, train_features, train_targets, test_features):
        """
        Preprocessing pipeline:
        (1) Fill in missing values
        (2) Detect and remove outliers from the train set
        (3) Rescale the features
        (4) Prune low-variance and correlated features
        (5) Feature selection
        """
        train_features = copy.deepcopy(train_features)
        train_targets = copy.deepcopy(train_targets)
        test_features = copy.deepcopy(test_features)
        imputer = copy.deepcopy(self.imputer)
        outliers_detector = copy.deepcopy(self.outliers_detector)
        scaler = copy.deepcopy(self.scaler)
        feature_selector = copy.deepcopy(self.feature_selector)

        # 1. Handling missing values
        self._count_missing_values(train_features)
        self._count_missing_values(test_features)
        self._print("Imputing missing values...")
        imputer.fit(train_features)
        train_features = imputer.transform(train_features)
        test_features = imputer.transform(test_features)
        assert self._count_missing_values(train_features) == 0.0
        assert self._count_missing_values(test_features) == 0.0

        # 2. Detect and remove outliers from the train set
        outliers = outliers_detector.fit_predict(train_features)
        outliers_count = sum(outliers == -1)
        outliers_pct = outliers_count / train_features.shape[0]
        self._print(f"{outliers_count} outliers detected! ({outliers_pct*100:.3f}%)")
        self._print("Removing outliers...")
        train_features = train_features[outliers == 1]
        train_targets = train_targets[outliers == 1]
        self._print(
            "Train shape:",
            train_features.shape,
            "Train target shape:",
            train_targets.shape,
        )

        # 3. Rescale the features
        self._print("Rescaling data...")
        scaler.fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        # 4. Prune low-variance and correlated features
        self._print("Removing zero-variance features...")
        self.var_selector = VarianceThreshold(threshold=0)
        self.var_selector.fit(train_features)
        train_features = self.var_selector.transform(train_features)
        test_features = self.var_selector.transform(test_features)
        self._print(
            "Train shape:", train_features.shape, "Test shape:", test_features.shape
        )
        self._print("Removing highly correlated features...")
        cor = self._corr2_coeff(train_features.T, train_features.T)
        correlated_pairs = np.argwhere(
            np.triu(np.absolute(cor) >= self.corr_threshold, 1)
        )
        train_features = np.delete(train_features, correlated_pairs[:, 1], axis=1)
        test_features = np.delete(test_features, correlated_pairs[:, 1], axis=1)
        self._print(
            "Train shape:", train_features.shape, "Test shape:", test_features.shape
        )

        # 5. Feature selection
        self._print("Selecting features...")
        feature_selector.fit(train_features, train_targets)
        train_features = feature_selector.transform(train_features)
        test_features = feature_selector.transform(test_features)
        self._print(
            "Train shape:", train_features.shape, "Test shape:", test_features.shape
        )

        return train_features, train_targets, test_features

    def _round_to_nearest_int(self, x):
        decimal = x % 1
        mask = abs(x - np.round(x)) <= self.rounding_threshold
        x[mask] = np.round(x[mask])
        return x

    def _cross_validation(self, model):
        scores = []
        self.oof_preds = []
        self.oof_targets = []
        for fold in self.folds:
            temp_model = copy.deepcopy(model)
            temp_model.fit(fold["x_train"], fold["y_train"])
            pred_val = temp_model.predict(fold["x_val"])

            if self.rounding_threshold is not None:
                pred_val = self._round_to_nearest_int(pred_val)

            r2 = r2_score(fold["y_val"], pred_val)
            scores.append(r2)
            self.oof_preds.extend(pred_val.tolist())
            self.oof_targets.extend(fold["y_val"].tolist())

        return scores

    def preprocessing(self):
        self.folds = []
        for fold_nr in range(5):
            self._print("### Preprocessing fold:", fold_nr, "###")
            val_mask = self.train_folds == fold_nr
            x_train = self.train_features[~val_mask]
            y_train = self.train_targets[~val_mask]
            x_val = self.train_features[val_mask]
            y_val = self.train_targets[val_mask]

            # Preprocess the current fold's train set
            x_train, y_train, x_val = self._preprocess_data(x_train, y_train, x_val)
            assert x_val.shape[0] == y_val.shape[0]
            self.folds.append(
                {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_val": x_val,
                    "y_val": y_val,
                }
            )

    def _save_oof_prediction(self):
        oof = pd.DataFrame()
        oof["y_pred"] = self.oof_preds
        oof["y"] = self.oof_targets
        oof.to_csv(f"submissions/{self.timestamp}.oof.csv", index=False)

    def train(self, model, rounding_threshold=None):
        self.timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        self.rounding_threshold = rounding_threshold
        scores = self._cross_validation(model)

        self._print("R2: {:.5f} ± {:.5f}".format(np.mean(scores), np.std(scores)))

        return np.mean(scores)

    def predict_test(self, model):
        # Run preprocessing on the entire train set
        train_features, train_targets, test_features = self._preprocess_data(
            self.train_features, self.train_targets, self.test_features
        )

        model.fit(train_features, train_targets)
        pred = model.predict(test_features)

        if self.rounding_threshold is not None:
            pred = self._round_to_nearest_int(pred)

        submission = pd.DataFrame()
        submission["id"] = self.test_df["id"]
        submission["y"] = pred
        submission.to_csv(f"submissions/last_submission.csv", index=False)
        self._save_oof_prediction()


SEED = 42


class TaskWrapper:
    def tune_solver(self, solver_params, verbose=False):
        imputer = solver_params["imputer"]
        imputer = globals()[imputer["name"]](**imputer["params"])

        outlier_detector = solver_params["outlier_detector"]
        outlier_detector = globals()[outlier_detector["name"]](
            **outlier_detector["params"]
        )

        scaler = solver_params["scaler"]
        scaler = globals()[scaler["name"]](**scaler["params"])

        feature_selector = solver_params["feature_selector"]
        feature_selector_estimator = globals()[feature_selector["name"]](
            **feature_selector["params"]
        )

        self.solver = TaskSolver(
            train_df=pd.read_csv("data/kfold.csv"),
            test_df=pd.read_csv("data/X_test.csv"),
            imputer=imputer,
            outliers_detector=outlier_detector,
            scaler=scaler,
            corr_threshold=solver_params["corr_threshold"],
            feature_selector=SelectFromModel(
                feature_selector_estimator, threshold=feature_selector["threshold"]
            ),
            verbose=verbose,
        )

        self.solver.preprocessing()

    def train_model(self, model_params):
        if model_params["name"] == "StackingRegressor":
            estimators = []
            i = 1
            for estimator in model_params["estimators"]:
                estimators.append(
                    (
                        estimator["name"] + str(i),
                        globals()[estimator["name"]](**estimator["params"]),
                    )
                )
                i += 1

            final_estimator = model_params["final_estimator"]
            final_estimator = globals()[final_estimator["name"]](
                **final_estimator["params"]
            )
            self.model = StackingRegressor(
                estimators=estimators, final_estimator=final_estimator
            )

        return self.solver.train(self.model)


solver_params = {
    "imputer": {
        "name": "IterativeImputer",
        "params": {
            "max_iter": 20,
            "n_nearest_features": 32,
            "random_state": SEED,
            "verbose": 0,
        },
    },
    "outlier_detector": {
        "name": "LocalOutlierFactor",
        "params": {
            "n_neighbors": 110,
            "algorithm": "auto",
            "contamination": "auto",
        },
    },
    "scaler": {
        "name": "QuantileTransformer",
        "params": {
            "output_distribution": "normal",
            "n_quantiles": 270,
            "random_state": SEED,
        },
    },
    "corr_threshold": 0.855019157169521,
    "feature_selector": {
        "name": "RandomForestRegressor",
        "params": {
            "bootstrap": False,
            "max_depth": 33,
            "max_features": "log2",
            "min_samples_leaf": 6,
            "min_samples_split": 21,
            "n_estimators": 794,
            "random_state": SEED,
        },
        "threshold": "mean",
    },
}

wrapper = TaskWrapper()
wrapper.tune_solver(solver_params, True)


def objective(trial):
    model_params = {
        "name": "StackingRegressor",
        "estimators": [
            {
                "name": "Ridge",
                "params": {
                    "alpha": trial.suggest_float("ridge_alpha", 1e-3, 100),
                    "random_state": SEED,
                },
            },
            {
                "name": "SVR",
                "params": {
                    "C": trial.suggest_float("C", 1.0, 50.0),
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("matern_length_scale1", 1, 20),
                        nu=1.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("matern_length_scale2", 1, 20),
                        nu=1.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("matern_length_scale3", 1, 20),
                        nu=1.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("matern_length_scale4", 1, 20),
                        nu=1.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("matern_length_scale5", 1, 20),
                        nu=1.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": RBF(
                        length_scale=trial.suggest_float("rbf_length_scale1", 1, 12.5)
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": RBF(
                        length_scale=trial.suggest_float("rbf_length_scale2", 1, 12.5)
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": RBF(
                        length_scale=trial.suggest_float("rbf_length_scale3", 1, 12.5)
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": RBF(
                        length_scale=trial.suggest_float("rbf_length_scale4", 1, 12.5)
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": RBF(
                        length_scale=trial.suggest_float("rbf_length_scale5", 1, 12.5)
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("exp_length_scale1", 1, 63),
                        nu=0.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("exp_length_scale2", 1, 63),
                        nu=0.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("exp_length_scale3", 1, 63),
                        nu=0.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("exp_length_scale4", 1, 63),
                        nu=0.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "GaussianProcessRegressor",
                "params": {
                    "normalize_y": True,
                    "kernel": Matern(
                        length_scale=trial.suggest_float("exp_length_scale5", 1, 63),
                        nu=0.5,
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "ExtraTreesRegressor",
                "params": {
                    "n_estimators": trial.suggest_int("etr_n_estimators", 50, 300),
                    "min_samples_split": trial.suggest_int(
                        "etr_min_samples_split", 2, 8
                    ),
                    "min_samples_leaf": trial.suggest_int("etr_min_samples_leaf", 1, 8),
                    "random_state": SEED,
                },
            },
            {
                "name": "XGBRegressor",
                "params": {
                    "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0),
                    "reg_alpha": trial.suggest_float("xgb_alpha", 1e-3, 10.0),
                    "colsample_bytree": trial.suggest_float(
                        "xgb_colsample_bytree", 0.5, 1
                    ),
                    "subsample": trial.suggest_float("xgb_subsample", 0.5, 1),
                    "learning_rate": trial.suggest_float(
                        "xgb_learning_rate", 0.008, 0.02
                    ),
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("xgb_max_depth", 4, 20),
                    "min_child_weight": trial.suggest_int(
                        "xgb_min_child_weight", 1, 300
                    ),
                    "random_state": SEED,
                },
            },
            {
                "name": "AdaBoostRegressor",
                "params": {
                    "n_estimators": trial.suggest_int("ada_n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("ada_learning_rate", 0.5, 2),
                    "random_state": SEED,
                },
            },
            {
                "name": "KNeighborsRegressor",
                "params": {
                    "n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
                    "weights": trial.suggest_categorical(
                        "kn_weights", ["uniform", "distance"]
                    ),
                },
            },
            {
                "name": "GradientBoostingRegressor",
                "params": {
                    "n_estimators": trial.suggest_int("gb_n_estimators", 50, 400),
                    "subsample": trial.suggest_float("gb_subsample", 0.5, 1),
                    "min_samples_split": trial.suggest_int(
                        "gb_min_samples_split", 2, 8
                    ),
                    "min_samples_leaf": trial.suggest_int("gb_min_samples_leaf", 2, 8),
                    "max_depth": trial.suggest_int("gb_max_depth", 2, 5),
                    "random_state": SEED,
                },
            },
            {
                "name": "RandomForestRegressor",
                "params": {
                    "n_estimators": trial.suggest_int("rf_n_estimators", 50, 400),
                    "min_samples_split": trial.suggest_int(
                        "rf_min_samples_split", 2, 8
                    ),
                    "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 2, 8),
                    "random_state": SEED,
                },
            },
        ],
        "final_estimator": {
            "name": "Ridge",
            "params": {
                "alpha": trial.suggest_float("stack_alpha", 1e-3, 100),
                "random_state": SEED,
            },
        },
    }

    cv_score = wrapper.train_model(model_params)
    return cv_score


study = optuna.create_study(
    direction="maximize",
    # sampler=optuna.samplers.TPESampler(seed=SEED),
    # pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

study.optimize(objective, n_trials=10000, n_jobs=4)
