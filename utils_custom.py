import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from scipy.stats import wilcoxon
import random
from sklearn.base import BaseEstimator, TransformerMixin

random.seed(42)
np.random.seed(42)


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
    pipe,
    pipe_grid,
    X,
    y,
    subsample=False,
    n_max=20_000,
    best_params=True,
    n_jobs=-1,
    cv=3,
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
