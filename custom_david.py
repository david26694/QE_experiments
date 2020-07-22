#!/usr/bin/env python
# coding: utf-8

# #### Libraries
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
import os
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from scipy.stats import wilcoxon

from sktools import QuantileEncoder, TypeSelector
from lightgbm import LGBMRegressor
from category_encoders.m_estimate import MEstimateEncoder
from tabulate import tabulate
from pathlib import Path

from utils import elapsed_time_mins, fit_pipe, compare_results, SummaryEncoder
from constants import data, keep, drop, cols_enc, target, columns

random.seed(42)
np.random.seed(42)

pd.set_option("max_columns", None)

# Check directories
for directory in [
    "./results_regression/pickle",
    "./results_regression/grid_results/",
    "./results_regression/partial/",
    "./results_regression/datasets/",
]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# # Loop

resultados = []

print(
    tabulate(
        tabular_data=[],
        headers=["Data", "Model", "Train", "Cv", "Test"],
        tablefmt="psql",
    )
)


results_dict = {}

for i, data_i in enumerate(data):

    dataset_keep = keep[i] + target[i]

    tic = time.time()

    cv = RepeatedKFold(n_repeats=3, n_splits=4)

    # Read data
    df = pd.read_csv(data_i)

    # if df.shape[0] > 100_000:
    #     df = df.sample(n=100_000)

    # Drop columns
    df = df.loc[:, dataset_keep]

    # Fillna
    df.fillna(0, inplace=True)

    print(df.shape)
    # Train-Test Split
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=target[i]), df[target[i]])

    results_dict[data_i] = {}

    # Elastic Net + target encoding
    scaler = StandardScaler()
    lm = ElasticNet()
    lgbm = LGBMRegressor(verbose=-1)
    te = MEstimateEncoder(cols=cols_enc[i])
    pe = QuantileEncoder(cols=cols_enc[i], quantile=0.50)
    se = SummaryEncoder(cols=cols_enc[i], quantiles=[0.25, 0.50, 0.75], m=100)

    encoders = {"te": te, "pe": pe, "se": se}
    learners = {"lm": lm, "lg": lgbm}

    for learner_name, learner in learners.items():

        results_dict[data_i][learner_name] = {}

        for encoder_name, encoder in encoders.items():
            results_dict[data_i][learner_name][encoder_name] = {}

            pipe = Pipeline(
                [
                    ("enc", encoder),
                    (
                        "selector",
                        TypeSelector(np.number),
                    ),  # Selects Numerical Columns only
                    ("scaler", scaler),
                    ("learner", learner),
                ]
            )

            pipe_grid = {}

            # Train model
            enet_te, enet_te_grid_results, enet_te_params = fit_pipe(
                pipe, pipe_grid, X_tr, y_tr, n_jobs=-1, cv=cv
            )

            results_dict[data_i][learner_name][encoder_name][
                "grid_results"
            ] = enet_te_grid_results

            results_dict[data_i][learner_name][encoder_name][
                "cv_mae"
            ] = -enet_te_grid_results["mean_test_score"]

            results_dict[data_i][learner_name][encoder_name][
                "train_mae"
            ] = mean_absolute_error(y_tr, enet_te.predict(X_tr))
            results_dict[data_i][learner_name][encoder_name][
                "test_mae"
            ] = mean_absolute_error(y_te, enet_te.predict(X_te))

            results_dict[data_i][learner_name][encoder_name][
                "train_mse"
            ] = mean_squared_error(y_tr, enet_te.predict(X_tr))
            results_dict[data_i][learner_name][encoder_name][
                "test_mse"
            ] = mean_squared_error(y_te, enet_te.predict(X_te))

            cv_results = results_dict[data_i][learner_name][encoder_name][
                "grid_results"
            ]

            cv_results['learner'] = learner_name
            cv_results['encoder'] = encoder_name
            cv_results['data'] = data_i

            csv_file = Path('results_regression', 'cv_results.csv')

            if csv_file.exists():
                cv_results.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                cv_results.to_csv(csv_file, index=False)

            print(
                tabulate(
                    tabular_data=[
                        [
                            data_i[5:10],
                            f"{learner_name}_{encoder_name}",
                            results_dict[data_i][learner_name][encoder_name][
                                "train_mae"
                            ],
                            results_dict[data_i][learner_name][encoder_name]["cv_mae"],
                            results_dict[data_i][learner_name][encoder_name][
                                "test_mae"
                            ],
                        ]
                    ],
                    tablefmt="psql",
                )
            )

            pd.DataFrame(
                results_dict[data_i][learner_name][encoder_name]["grid_results"]
            ).to_csv(
                "./results_regression/grid_results/{}_{}.csv".format(
                    f"{learner_name}_{encoder_name}_grid_results", data_i[5:10]
                )
            )

            if encoder_name != "te":

                pvalue = compare_results(
                    results_dict[data_i][learner_name]["te"]["grid_results"],
                    results_dict[data_i][learner_name][encoder_name]["grid_results"],
                )
                print(f"p-value of wilcoxon test with {encoder_name}: {pvalue}")

    # Add Results
    resultados.append(
        [
            data_i.split("/")[1],
            # Scores
            results_dict[data_i]["lm"]["te"]["train_mae"],
            results_dict[data_i]["lm"]["te"]["test_mae"],
            results_dict[data_i]["lm"]["te"]["train_mse"],
            results_dict[data_i]["lm"]["te"]["test_mse"],
            results_dict[data_i]["lm"]["pe"]["train_mae"],
            results_dict[data_i]["lm"]["pe"]["test_mae"],
            results_dict[data_i]["lm"]["pe"]["train_mse"],
            results_dict[data_i]["lm"]["pe"]["test_mse"],
            results_dict[data_i]["lg"]["te"]["train_mae"],
            results_dict[data_i]["lg"]["te"]["test_mae"],
            results_dict[data_i]["lg"]["te"]["train_mse"],
            results_dict[data_i]["lg"]["te"]["test_mse"],
            results_dict[data_i]["lg"]["pe"]["train_mae"],
            results_dict[data_i]["lg"]["pe"]["test_mae"],
            results_dict[data_i]["lg"]["pe"]["train_mse"],
            results_dict[data_i]["lg"]["pe"]["test_mse"],
            # Shape
            df.shape,
            # params
            "",
            "",
            # Time
            elapsed_time_mins(tic, time.time()),
        ]
    )


resultados = pd.DataFrame(resultados, columns=columns)
resultados.to_csv("./results_regression/resultados.csv", index=False)
