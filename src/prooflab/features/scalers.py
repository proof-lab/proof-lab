"""Leak-free feature scalers with strict fit/transform separation.

Ensures scaling statistics (mean, variance, median, IQR, min/max) are computed
strictly on training splits and applied downstream without look-ahead data leakage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class NotFittedError(ValueError):
    """Raised when transform or inverse_transform is called before fitting."""


class BaseScaler(ABC):
    """Abstract base class for all leak-free tabular feature scalers."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        """Fit scaling statistics strictly on the provided training DataFrame."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned scaling statistics to transform the provided DataFrame."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit to training data, then transform it."""
        return self.fit(df).transform(df)

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revert transformed data back to original scale."""


class StandardScaler(BaseScaler):
    """Standardizes features by removing the mean and scaling to unit variance (Z-score)."""

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__(columns=columns)
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        if self.columns is not None:
            target_cols = self.columns
        else:
            target_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.means_.clear()
        self.stds_.clear()

        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame to fit.")
            mean_val = float(df[col].mean())
            std_val = float(df[col].std(ddof=0))
            # Protect against division by zero for constant features
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0
            self.means_[col] = mean_val
            self.stds_[col] = std_val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError("StandardScaler is not fitted yet. Call 'fit' before 'transform'.")

        result = df.copy()
        for col, mean_val in self.means_.items():
            if col in result.columns:
                std_val = self.stds_[col]
                result[col] = (result[col].astype(float) - mean_val) / std_val
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(
                "StandardScaler is not fitted yet. Call 'fit' before 'inverse_transform'."
            )

        result = df.copy()
        for col, mean_val in self.means_.items():
            if col in result.columns:
                std_val = self.stds_[col]
                result[col] = (result[col].astype(float) * std_val) + mean_val
        return result


class RobustScaler(BaseScaler):
    """Scales features using median and IQR for robustness against outliers."""

    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__(columns=columns)
        self.medians_: dict[str, float] = {}
        self.iqrs_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        if self.columns is not None:
            target_cols = self.columns
        else:
            target_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.medians_.clear()
        self.iqrs_.clear()

        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame to fit.")
            med = float(df[col].median())
            q25 = float(df[col].quantile(0.25))
            q75 = float(df[col].quantile(0.75))
            iqr = q75 - q25
            if iqr == 0.0 or np.isnan(iqr):
                iqr = 1.0
            self.medians_[col] = med
            self.iqrs_[col] = iqr

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError("RobustScaler is not fitted yet. Call 'fit' before 'transform'.")

        result = df.copy()
        for col, med in self.medians_.items():
            if col in result.columns:
                iqr = self.iqrs_[col]
                result[col] = (result[col].astype(float) - med) / iqr
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(
                "RobustScaler is not fitted yet. Call 'fit' before 'inverse_transform'."
            )

        result = df.copy()
        for col, med in self.medians_.items():
            if col in result.columns:
                iqr = self.iqrs_[col]
                result[col] = (result[col].astype(float) * iqr) + med
        return result


class MinMaxScaler(BaseScaler):
    """Transforms features by scaling each feature to a given range (default [0, 1])."""

    def __init__(
        self,
        columns: list[str] | None = None,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__(columns=columns)
        self.feature_range = feature_range
        self.mins_: dict[str, float] = {}
        self.maxs_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        if self.columns is not None:
            target_cols = self.columns
        else:
            target_cols = list(df.select_dtypes(include=[np.number]).columns)
        self.mins_.clear()
        self.maxs_.clear()

        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame to fit.")
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if max_val == min_val:
                max_val = min_val + 1.0
            self.mins_[col] = min_val
            self.maxs_[col] = max_val

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError("MinMaxScaler is not fitted yet. Call 'fit' before 'transform'.")

        result = df.copy()
        f_min, f_max = self.feature_range
        for col, min_val in self.mins_.items():
            if col in result.columns:
                max_val = self.maxs_[col]
                norm = (result[col].astype(float) - min_val) / (max_val - min_val)
                result[col] = (norm * (f_max - f_min)) + f_min
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(
                "MinMaxScaler is not fitted yet. Call 'fit' before 'inverse_transform'."
            )

        result = df.copy()
        f_min, f_max = self.feature_range
        for col, min_val in self.mins_.items():
            if col in result.columns:
                max_val = self.maxs_[col]
                scaled = (result[col].astype(float) - f_min) / (f_max - f_min)
                result[col] = (scaled * (max_val - min_val)) + min_val
        return result
