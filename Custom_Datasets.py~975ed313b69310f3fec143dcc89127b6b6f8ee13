#!/usr/bin/env python
# coding: utf-8

# #### Libraries
import sklearn

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

pd.set_option("max_columns", None)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from scipy.stats import wilcoxon


from sklearn.feature_selection import VarianceThreshold
import os


from lightgbm import LGBMRegressor
import time

from category_encoders.m_estimate import MEstimateEncoder

import warnings

warnings.filterwarnings("ignore")

import sktools
from utils_custom import *
from tabulate import tabulate


# Check directories
directory = "./results_regression/pickle"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = "./results_regression/grid_results/"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = "./results_regression/partial/"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = "./results_regression/datasets/"
if not os.path.exists(directory):
    os.makedirs(directory)

data = [
    "data/house_kaggle.csv",
    "data/stackoverflow.csv",
    "data/so2019.csv",
    "data/ks.csv",
    "data/medical_payments_sample.csv",
    "data/cauchy.csv",
]


drop = [
    [
        "Id",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinSF1",
        "BsmtFinType2",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "LowQualFinSF",
        "FullBath",
        "HalfBath",
    ],
    ["Respondent", "Salary"],
    [],
    [],
    [],
    [],
]


cols_enc = [
    [
        "MSSubClass",
        "MSZoning",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "Neighborhood",
        "BldgType",
        "HouseStyle",
        "YearBuilt",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "ExterQual",
        "MasVnrType",
        "Heating",
        "HeatingQC",
    ],
    [
        "Country",
        "Employment",
        "FormalEducation",
        "UndergradMajor",
        "CompanySize",
        "DevType",
        "YearsCoding",
        "LanguageWorkedWith",
        "LanguageDesireNextYear",
        "RaceEthnicity",
    ],
    ["yearscode", "country"],
    ["category", "main_category", "currency", "state", "country"],
    [
        "Recipient_City",
        "Recipient_State",
        "Recipient_Zip_Code",
        "Recipient_Country",
        "Physician_Primary_Type",
        "Physician_License_State_code1",
        "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name",
        "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Country",
        "Form_of_Payment_or_Transfer_of_Value",
        "Nature_of_Payment_or_Transfer_of_Value",
    ],
    ["value_1", "value_2"],
]


target = [
    ["SalePrice"],
    ["ConvertedSalary"],
    ["convertedcomp"],
    ["goal"],
    ["Total_Amount_of_Payment_USDollars"],
    ["target"],
]


# # Loop

n_jobs = 1
float_eltype = np.float32
resultados = []
tic = time.time()

n_max = 20_000
cv = 4
filter_size = 2_000
columns = [
    "NameDataset",
    # Scores
    "enet_te_train_mae",
    "enet_te_test_mae",
    "enet_te_train_mse",
    "enet_te_test_mse",
    "enet_pe_train_mae",
    "enet_pe_test_mae",
    "enet_pe_train_mse",
    "enet_pe_test_mse",
    "xgb_te_train_mae",
    "xgb_te_test_mae",
    "xgb_te_train_mse",
    "xgb_te_test_mse",
    "xgb_pe_train_mae",
    "xgb_pe_test_mae",
    "xgb_pe_train_mse",
    "xgb_pe_test_mse",
    "size",
    # Params
    "enet_te_best_params",
    "enet_pe_best_params",
    # Time
    "time_train_m",
]


print(
    tabulate(
        tabular_data=[],
        headers=["Data", "Model", "Train", "Cv", "Test"],
        tablefmt="psql",
    )
)

results_dict = {}

for i, data_i in enumerate(data):

    cv = RepeatedKFold(n_repeats=3, n_splits=4)

    # Read data
    df = pd.read_csv(data_i)

    if df.shape[0] > 100_000:
        df = df.sample(n=100_000)

    # Drop columns
    df = df.drop(columns=drop[i])

    # Fillna
    df.fillna(0, inplace=True)

    print(df.shape)
    # Train-Test Split
    X_tr, X_te, y_tr, y_te = sklearn.model_selection.train_test_split(
        df.drop(columns=target[i]), df[target[i]]
    )

    results_dict[data_i] = {}

    # Elastic Net + target encoding
    scaler = sklearn.preprocessing.StandardScaler()
    lm = sklearn.linear_model.ElasticNet()
    lgbm = LGBMRegressor()
    te = MEstimateEncoder(cols=cols_enc[i])
    pe = sktools.QuantileEncoder(cols=cols_enc[i], quantile=0.50)

    encoders = {"te": te, "pe": pe}
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
                pipe, pipe_grid, X_tr, y_tr, n_jobs=n_jobs, cv=cv
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

            print(
                tabulate(
                    tabular_data=[
                        [
                            data[i][5:10],
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
                    f"{learner_name}_{encoder_name}_grid_results", data[i][5:10]
                )
            )

        pvalue = compare_results(
            results_dict[data_i][learner_name]["te"]["grid_results"],
            results_dict[data_i][learner_name]["pe"]["grid_results"],
        )
        print(f"p-value of wilcoxon test {pvalue}")

    # Add Results
    resultados.append(
        [
            data[i].split("/")[1],
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
