# simple-tabular
*A quick classification/regression ML pipeline for tabular data.*

This template stems from my winning project code for the Advanced Machine Learning course at ETH Zurich.
The entire logic of this pipeline is contained in the `TaskSolver` class, featuring the several standard steps:

1. Preprocessing step consists of: data imputation, outliers detection and removal heuristics, data rescaling, redundant features pruning.
2. Feature selection.
3. Model fitting.

To avoid leakage, the steps above are carried per fold instead on the entire train set.
Furthermore, at the outer level, I employ Optuna for hyperparameters tuning and pass the parameters to the `TaskSolver` object.

An example of the hyperparameter setting:

```python
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
```

For the modeling step, I use stacked generalization to maximize the predictive power and also to make a good use of the distributbed tuning in Optuna.

```python
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
```