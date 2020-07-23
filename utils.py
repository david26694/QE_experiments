import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator
from category_encoders.utils import TransformerWithTargetMixin
from sktools import QuantileEncoder


def elapsed_time_mins(time1, time2):
    elapsed = np.round(np.abs(time1 - time2) / 60, decimals=2)

    return elapsed


def fit_pipe(
    pipe,
    pipe_grid,
    X,
    y,
    subsample=False,
    n_max=20_000,
    best_params=True,
    n_jobs=-1,
    cv=5,
):
    y = np.ravel(y.copy())
    if subsample:
        X_sampled = X[0:n_max]
        y_sampled = y[0:n_max]
    else:
        X_sampled = X
        y_sampled = y

    # Instantiate the grid
    pipe_cv = GridSearchCV(
        pipe,
        param_grid=pipe_grid,
        n_jobs=n_jobs,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    pipe_cv.fit(X_sampled, y_sampled)

    best_estimator = pipe_cv.best_estimator_.fit(X, y)
    grid_results = pd.DataFrame(pipe_cv.cv_results_)

    return best_estimator, grid_results, pipe_cv.best_params_


def compare_results(grid_1_res, grid_2_res):
    all_results = grid_1_res.melt().merge(
        grid_2_res.melt(), on="variable", suffixes=("_te", "_pe")
    )

    all_results = all_results[all_results["variable"].str.contains("split")]

    test_results = wilcoxon(
        -all_results.value_pe, -all_results.value_te, alternative="less"
    )

    return test_results.pvalue.round(3)


class SummaryEncoder(BaseEstimator, TransformerWithTargetMixin):
    def __init__(self, cols, quantiles, m=1.0):

        self.cols = cols
        self.quantiles = quantiles
        self.m = m
        self.encoder_list = None

    def fit(self, X, y):

        X = X.copy()

        for quantile in self.quantiles:
            for col in self.cols:
                percentile = round(quantile * 100)
                X[f"{col}_{percentile}"] = X[col]

        encoder_list = []
        for quantile in self.quantiles:
            col_names = []
            for col in self.cols:
                percentile = round(quantile * 100)
                col_names.append(f"{col}_{percentile}")
            enc = QuantileEncoder(cols=col_names, quantile=quantile, m=self.m)
            enc.fit(X, y)
            encoder_list.append(enc)

        self.encoder_list = encoder_list

        return self

    def transform(self, X, y=None):
        X_encoded = X.copy()

        for quantile in self.quantiles:
            for col in self.cols:
                percentile = round(quantile * 100)
                X_encoded[f"{col}_{percentile}"] = X_encoded[col]

        for encoder in self.encoder_list:
            X_encoded = encoder.transform(X_encoded)
        return X_encoded
