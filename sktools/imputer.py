"""Main module."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IsEmptyExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that adds columns indicating wether columns have NaN values
    in a row
    """

    def __init__(self, keep_trivial=False, selected_columns=None):
        """

        @param keep_trivial: If a column doesn't have NaN, don't add the
        @param selected_columns: List of columns to transform. If None,
        all columns are transformed. It only makes sense in the data frame case

        column
        """
        self.keep_trivial = keep_trivial
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        return self

    def transform_data_frame(self, X):
        """
        Transform method in case of receiving a pandas data frame

        @param X: pd.DataFrame to transform
        @return: Transformed data frame
        """

        new_x = X.copy()

        if self.selected_columns is None:
            selected_columns = X.columns
        else:
            selected_columns = self.selected_columns

        for column in selected_columns:

            new_column_name = f"{column}_na"

            is_na = X[column].isna()

            if any(is_na == True) or self.keep_trivial:
                new_x[new_column_name] = is_na

        return new_x

    def transform(self, X):
        """
        For each column, it creates a new one indicating if that column is na

        @param X: pd.DataFrame or numpy array to transform
        @return: New object with more columns in case there are NA
        """

        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)

        if isinstance(X, pd.DataFrame):
            return self.transform_data_frame(X)

        if isinstance(X, np.ndarray):
            new_x = pd.DataFrame(X.copy())
            transformed_x = self.transform_data_frame(new_x)
            return transformed_x.values
