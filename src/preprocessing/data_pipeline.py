"""
SO-SAFED Data Preprocessing Pipeline

Handles loading, concatenation, and preparation of hourly PED admission
data for TiDE-RIN forecasting. The pipeline:

1. Loads historical admission data from CSV
2. Aggregates to hourly resolution
3. Validates stationarity (ADF test)
4. Creates Darts TimeSeries objects
5. Generates temporal covariates (hour, day, week, month, year)
6. Scales data for model input
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr


class DataPipeline:
    """Prepares PED admission data for TiDE-RIN forecasting."""

    def __init__(self, data_path: str, date_col: str = "date", value_col: str = "apply_number"):
        self.data_path = Path(data_path)
        self.date_col = date_col
        self.value_col = value_col
        self.scaler = Scaler()
        self._df = None
        self._series = None
        self._scaled_series = None
        self._covariates = None

    def load(self) -> pd.DataFrame:
        """Load and parse the admission dataset."""
        df = pd.read_csv(self.data_path)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.reset_index(drop=True)
        self._df = df
        return df

    def validate_stationarity(self, significance: float = 0.05) -> dict:
        """Run Augmented Dickey-Fuller test on the time series.

        Returns
        -------
        dict
            ADF test statistic, p-value, and whether the series is stationary.
        """
        if self._df is None:
            raise ValueError("Call load() first.")

        from statsmodels.tsa.stattools import adfuller

        result = adfuller(self._df[self.value_col].dropna())
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < significance,
            "critical_values": result[4],
        }

    def create_series(self, freq: str = "h") -> TimeSeries:
        """Convert DataFrame to Darts TimeSeries at the given frequency."""
        if self._df is None:
            raise ValueError("Call load() first.")

        self._series = TimeSeries.from_dataframe(
            df=self._df,
            time_col=self.date_col,
            value_cols=self.value_col,
            freq=freq,
        )
        return self._series

    def scale(self) -> TimeSeries:
        """Fit scaler on the series and return scaled version."""
        if self._series is None:
            raise ValueError("Call create_series() first.")

        self._scaled_series = self.scaler.fit_transform(self._series)
        return self._scaled_series

    def create_covariates(self) -> TimeSeries:
        """Generate temporal covariates for the TiDE model.

        Creates normalized cyclic features: hour, day, week, month, year.
        """
        if self._scaled_series is None:
            raise ValueError("Call scale() first.")

        year_min = self._df[self.date_col].min().year

        self._covariates = concatenate(
            [
                dt_attr(self._series.time_index, "hour", dtype=np.float32) / 24,
                dt_attr(self._scaled_series.time_index, "day", dtype=np.float32) / 30,
                dt_attr(self._scaled_series.time_index, "week", dtype=np.float32) / 7,
                dt_attr(self._scaled_series.time_index, "month", dtype=np.float32) / 12,
                (dt_attr(self._scaled_series.time_index, "year", dtype=np.float32) - year_min),
            ],
            axis="component",
        )
        return self._covariates

    def prepare(self, freq: str = "h") -> tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Run the full pipeline: load -> series -> scale -> covariates.

        Returns
        -------
        tuple
            (raw_series, scaled_series, covariates)
        """
        self.load()
        self.create_series(freq=freq)
        self.scale()
        self.create_covariates()
        return self._series, self._scaled_series, self._covariates

    def inverse_transform(self, prediction: TimeSeries) -> TimeSeries:
        """Inverse-scale a model prediction back to original units."""
        return self.scaler.inverse_transform(prediction)

    @staticmethod
    def aggregate_to_shift_blocks(
        df: pd.DataFrame, date_col: str = "date", value_col: str = "apply_number"
    ) -> pd.DataFrame:
        """Aggregate hourly data into 8-hour shift blocks (08-16, 16-24, 24-08).

        Used for comparing forecasted demand against shift capacity.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        def assign_shift(hour: int) -> str:
            if 8 <= hour < 16:
                return "08-16"
            elif 16 <= hour < 24:
                return "16-24"
            return "24-08"

        df["shift"] = df[date_col].dt.hour.map(assign_shift)
        df["day"] = df[date_col].dt.date

        grouped = df.groupby(["day", "shift"])[value_col].sum().reset_index()
        grouped.columns = ["date", "shift", "total_patients"]
        return grouped
