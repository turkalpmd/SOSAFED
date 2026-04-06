"""
Invertible Data Transformer Base Class

Adapted from the Darts library (https://github.com/unit8co/darts) for use in
the SO-SAFED preprocessing pipeline. Provides the base class for reversible
data transformations used in the time-series forecasting workflow.

License: Apache License 2.0 (Darts library)
"""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.base_data_transformer import (
    BaseDataTransformer,
    component_masking,
)
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply

logger = get_logger(__name__)


class InvertibleDataTransformer(BaseDataTransformer):
    def __init__(
        self,
        name: str = "InvertibleDataTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
        parallel_params: Union[bool, Sequence[str]] = False,
        mask_components: bool = True,
    ):
        """Abstract class for invertible transformers.

        Deriving classes must implement ts_transform() and ts_inverse_transform().
        Supports parallelised transformation across multiple TimeSeries.

        Parameters
        ----------
        name
            Transformer name.
        n_jobs
            Number of parallel jobs (-1 = all processors).
        verbose
            Whether to print progress.
        parallel_params
            Which fixed parameters vary across parallel jobs.
        mask_components
            Whether to automatically apply component masks.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=mask_components,
        )

    @classmethod
    @component_masking
    def _ts_inverse_transform(cls, *args, **kwargs):
        return cls.ts_inverse_transform(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any]
    ) -> TimeSeries:
        """Inverse-transform a single TimeSeries.

        Must undo the transformation performed by ts_transform().

        Parameters
        ----------
        series
            Series to inverse-transform.
        params
            Dictionary with 'fixed' and optionally 'fitted' parameters.
        """
        pass

    def inverse_transform(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]],
        *args,
        component_mask: Optional[np.array] = None,
        **kwargs,
    ) -> Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]]:
        """Inverse-transform a (sequence of) series.

        Handles parallelisation and component masking automatically.

        Parameters
        ----------
        series
            Single TimeSeries, sequence, or list of lists.
        component_mask
            Optional boolean array specifying which components to transform.

        Returns
        -------
        Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]
            Inverse-transformed data in the same structure as input.
        """
        if hasattr(self, "_fit_called") and not self._fit_called:
            raise_log(
                ValueError("fit() must have been called before inverse_transform()"),
                logger=logger,
            )

        desc = f"Inverse ({self._name})"

        called_with_single_series = False
        called_with_sequence_series = False
        if isinstance(series, TimeSeries):
            data = [series]
            transformer_selector = [0]
            called_with_single_series = True
        elif isinstance(series[0], TimeSeries):
            data = series
            transformer_selector = range(len(series))
            called_with_sequence_series = True
        else:
            data = []
            transformer_selector = []
            for idx, series_list in enumerate(series):
                data.extend(series_list)
                transformer_selector += [idx] * len(series_list)

        input_iterator = _build_tqdm_iterator(
            zip(data, self._get_params(transformer_selector=transformer_selector)),
            verbose=self._verbose,
            desc=desc,
            total=len(transformer_selector),
        )

        kwargs["mask_components"] = self._mask_components
        kwargs["mask_components_apply_only"] = False
        kwargs["component_mask"] = component_mask

        transformed_data = _parallel_apply(
            input_iterator,
            self._ts_inverse_transform,
            self._n_jobs,
            args,
            kwargs,
        )

        if called_with_single_series:
            return transformed_data[0]
        elif called_with_sequence_series:
            return transformed_data
        else:
            cum_len = np.cumsum([0] + [len(s_) for s_ in series])
            return [
                transformed_data[cum_len[i] : cum_len[i + 1]]
                for i in range(len(cum_len) - 1)
            ]
