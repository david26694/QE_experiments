#!/usr/bin/env python
# coding: utf-8

# #### Libraries
import sklearn

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

pd.set_option("max_columns", None)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from scipy.stats import wilcoxon


from sklearn.feature_selection import VarianceThreshold
import os


from lightgbm import LGBMRegressor


import random
random.seed(0)

import time

from category_encoders.m_estimate import MEstimateEncoder

import warnings

warnings.filterwarnings("ignore")

import sktools
from tabulate import tabulate


# In[27]:


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that filters a type of columns of a given data frame.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        # print("Type Selector out shape {}".format(X.select_dtypes(include=[self.dtype]).shape))
        # print(X.select_dtypes(include=[self.dtype]).dtypes)
        return X.select_dtypes(include=[self.dtype])


def elapsed_time_mins(time1, time2):
    elapsed = np.round(np.abs(time1 - time2) / 60, decimals=2)

    return elapsed


def fit_pipe(
    pipe, pipe_grid, X, y, subsample=False, n_max=20_000, best_params=True
):

    if subsample:
        X = X[0:n_max]
        y = y[0:n_max]

    # Instantiate the grid
    pipe_cv = GridSearchCV(
        pipe,
        param_grid=pipe_grid,
        n_jobs=n_jobs,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    pipe_cv.fit(X, y)

    best_estimator = pipe_cv.best_estimator_.fit(X_tr, y_tr)
    grid_results = pd.DataFrame(pipe_cv.cv_results_)

    return best_estimator, grid_results, pipe_cv.best_params_


def compare_results(grid_1_res, grid_2_res):

    all_results = grid_1_res.melt().merge(
        grid_2_res.melt(), on="variable", suffixes=("_te", "_pe")
    )

    all_results = all_results[all_results["variable"].str.contains("split")]

    test_results = wilcoxon(
        all_results.value_pe, all_results.value_te, alternative="greater"
    )

    return test_results.pvalue.round(3)

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
        headers=["Data", "Model", "Train", "Test", "pvalue"],
        tablefmt="psql",
    )
)

for i in range(0, len(data)):

    cv = RepeatedKFold(n_repeats=3, n_splits=4, random_state=42)

    # Read data
    df = pd.read_csv(data[i])

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

    # Elastic Net + target encoding
    scaler = sklearn.preprocessing.StandardScaler()
    clf = sklearn.linear_model.ElasticNet()
    te = MEstimateEncoder(cols=cols_enc[i])

    pipe = Pipeline(
        [
            ("te", te),
            (
                "selector",
                TypeSelector(np.number),
            ),  # Selects Numerical Columns only
            ("scaler", scaler),
            ("clf", clf),
        ]
    )

    pipe_grid = {
        "te__m": [1],
    }

    # Train model
    enet_te, enet_te_grid_results, enet_te_params = fit_pipe(
        pipe, pipe_grid, X_tr, y_tr
    )

    score_enet_te_train = mean_absolute_error(y_tr, enet_te.predict(X_tr))
    score_enet_te_test = mean_absolute_error(y_te, enet_te.predict(X_te))

    score_enet_te_train_mse = mean_squared_error(y_tr, enet_te.predict(X_tr))
    score_enet_te_test_mse = mean_squared_error(y_te, enet_te.predict(X_te))

    print(
        tabulate(
            tabular_data=[
                [
                    data[i][5:10],
                    "enet_te",
                    score_enet_te_train,
                    score_enet_te_test,
                    np.nan,
                ]
            ],
            tablefmt="psql",
        )
    )

    # Elastic Net + percentile encoding
    scaler = sklearn.preprocessing.StandardScaler()
    clf = sklearn.linear_model.ElasticNet()
    pe = sktools.QuantileEncoder(cols=cols_enc[i], quantile=0.50, m=0)

    pipe = Pipeline(
        [
            ("pe", pe),
            (
                "selector",
                TypeSelector(np.number),
            ),  # Selects Numerical Columns only
            ("scaler", scaler),
            ("clf", clf),
        ]
    )

    pipe_grid = {
        "pe__m": [1],
        "pe__quantile": [0.50],
    }

    # Train model
    enet_pe, enet_pe_grid_results, enet_pe_params = fit_pipe(
        pipe, pipe_grid, X_tr, y_tr
    )

    score_enet_pe_train = mean_absolute_error(y_tr, enet_pe.predict(X_tr))
    score_enet_pe_test = mean_absolute_error(y_te, enet_pe.predict(X_te))

    score_enet_pe_train_mse = mean_squared_error(y_tr, enet_pe.predict(X_tr))
    score_enet_pe_test_mse = mean_squared_error(y_te, enet_pe.predict(X_te))

    pvalue = compare_results(enet_te_grid_results, enet_pe_grid_results)
    print(
        tabulate(
            tabular_data=[
                [
                    data[i][5:10],
                    "enet_pe",
                    score_enet_pe_train,
                    score_enet_pe_test,
                    pvalue,
                ]
            ],
            tablefmt="psql",
        )
    )

    # xgb + target encoding
    scaler = sklearn.preprocessing.StandardScaler()
    clf = LGBMRegressor()
    te = MEstimateEncoder(cols=cols_enc[i])
    var = VarianceThreshold(threshold=0.1)

    pipe = Pipeline(
        [
            ("te", te),
            (
                "selector",
                TypeSelector(np.number),
            ),  # Selects Numerical Columns only
            ("var", var),
            ("scaler", scaler),
            ("clf", clf),
        ]
    )

    pipe_grid = {
        "te__m": [1],
    }

    # Train model
    xgb_te, xgb_te_grid_results, xgb_te_params = fit_pipe(
        pipe, pipe_grid, X_tr, y_tr
    )

    score_xgb_te_train = mean_absolute_error(y_tr, xgb_te.predict(X_tr))
    score_xgb_te_test = mean_absolute_error(y_te, xgb_te.predict(X_te))

    score_xgb_te_train_mse = mean_squared_error(y_tr, xgb_te.predict(X_tr))
    score_xgb_te_test_mse = mean_squared_error(y_te, xgb_te.predict(X_te))

    print(
        tabulate(
            tabular_data=[
                [
                    data[i][5:10],
                    "xgbs_te ",
                    score_xgb_te_train,
                    score_xgb_te_test,
                    np.nan,
                ]
            ],
            tablefmt="psql",
        )
    )

    # xgb + percentile encoding
    scaler = sklearn.preprocessing.StandardScaler()
    clf = LGBMRegressor()
    pe = sktools.QuantileEncoder(cols=cols_enc[i], quantile=0.5, m=0)
    var = VarianceThreshold(threshold=0.01)

    pipe = Pipeline(
        [
            ("pe", pe),
            (
                "selector",
                TypeSelector(np.number),
            ),  # Selects Numerical Columns only
            ("var", var),
            ("scaler", scaler),
            ("clf", clf),
        ]
    )

    pipe_grid = {
        "pe__m": [1],
        "pe__quantile": [0.50],
    }

    # Train model
    xgb_pe, xgb_pe_grid_results, xgb_pe_params = fit_pipe(
        pipe, pipe_grid, X_tr, y_tr
    )

    score_xgb_pe_train = mean_absolute_error(y_tr, xgb_pe.predict(X_tr))
    score_xgb_pe_test = mean_absolute_error(y_te, xgb_pe.predict(X_te))

    score_xgb_pe_train_mse = mean_squared_error(y_tr, xgb_pe.predict(X_tr))
    score_xgb_pe_test_mse = mean_squared_error(y_te, xgb_pe.predict(X_te))

    pvalue = compare_results(xgb_te_grid_results, xgb_pe_grid_results)
    print(
        tabulate(
            tabular_data=[
                [
                    data[i][5:10],
                    "xgbs_pe",
                    score_xgb_pe_train,
                    score_xgb_pe_test,
                    pvalue,
                ]
            ],
            tablefmt="psql",
        )
    )

    # Grid Results
    pd.DataFrame(enet_te_grid_results).to_csv(
        "./results_regression/grid_results/{}_{}.csv".format(
            "enet_te_grid_results", data[i][5:10]
        )
    )
    pd.DataFrame(enet_pe_grid_results).to_csv(
        "./results_regression/grid_results/{}_{}.csv".format(
            "enet_pe_grid_results", data[i][5:10]
        )
    )
    pd.DataFrame(xgb_te_grid_results).to_csv(
        "./results_regression/grid_results/{}_{}.csv".format(
            "xgb_te_grid_results", data[i][5:10]
        )
    )
    pd.DataFrame(xgb_pe_grid_results).to_csv(
        "./results_regression/grid_results/{}_{}.csv".format(
            "xgbt_pe_grid_results", data[i][5:10]
        )
    )

    # Add Results
    resultados.append(
        [
            data[i].split("/")[1],
            # Scores
            score_enet_te_train,
            score_enet_te_test,
            score_enet_te_train_mse,
            score_enet_te_test_mse,
            score_enet_pe_train,
            score_enet_pe_test,
            score_enet_pe_train_mse,
            score_enet_pe_test_mse,
            score_xgb_te_train,
            score_xgb_te_test,
            score_xgb_te_train_mse,
            score_xgb_te_test_mse,
            score_xgb_pe_train,
            score_xgb_pe_test,
            score_xgb_pe_train_mse,
            score_xgb_pe_test_mse,
            # Shape
            df.shape,
            # params
            enet_te_params,
            enet_pe_params,
            # Time
            elapsed_time_mins(tic, time.time()),
        ]
    )


resultados = pd.DataFrame(resultados, columns=columns)
resultados.to_csv("./results_regression/resultados.csv", index=False)
