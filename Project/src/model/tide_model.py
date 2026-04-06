"""
Time-series Dense Encoder (TiDE) with Reversible Instance Normalization

This module is adapted from the Darts library (https://github.com/unit8co/darts)
for use in the SO-SAFED system. The original implementation is based on:

    [1] A. Das et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder",
        http://arxiv.org/abs/2304.08424
    [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series
        Forecasting against Distribution Shift",
        https://openreview.net/forum?id=cGDAkQo1C0p

License: Apache License 2.0 (Darts library)
"""

from typing import Optional

import torch
import torch.nn as nn

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

logger = get_logger(__name__)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Residual Block from the TiDE paper with skip connection and optional LayerNorm."""
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            MonteCarloDropout(dropout),
        )

        self.skip = nn.Linear(input_dim, output_dim)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x) + self.skip(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class _TideModule(PLMixedCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_past: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        temporal_hidden_size_past: Optional[int] = None,
        temporal_hidden_size_future: Optional[int] = None,
        **kwargs,
    ):
        """TiDE architecture: MLP-based encoder-decoder without attention.

        Parameters
        ----------
        input_dim
            Number of input components (target + optional covariates).
        output_dim
            Number of output components in the target.
        future_cov_dim
            Number of future covariates.
        static_cov_dim
            Number of static covariates.
        nr_params
            Number of parameters of the likelihood (or 1 if no likelihood).
        num_encoder_layers
            Number of stacked Residual Blocks in the encoder.
        num_decoder_layers
            Number of stacked Residual Blocks in the decoder.
        decoder_output_dim
            Dimensionality of the decoder output.
        hidden_size
            Width of hidden layers in encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            Width of hidden layers in the temporal decoder.
        temporal_width_past
            Width of past covariate embedding space.
        temporal_width_future
            Width of future covariate embedding space.
        use_layer_norm
            Whether to use layer normalization.
        dropout
            Dropout probability (compatible with MC dropout at inference).
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.temporal_hidden_size_past = temporal_hidden_size_past or hidden_size
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size

        # Past covariates handling
        self.past_cov_projection = None
        if self.past_cov_dim and temporal_width_past:
            self.past_cov_projection = _ResidualBlock(
                input_dim=self.past_cov_dim,
                output_dim=temporal_width_past,
                hidden_size=temporal_hidden_size_past,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            past_covariates_flat_dim = self.input_chunk_length * temporal_width_past
        elif self.past_cov_dim:
            past_covariates_flat_dim = self.input_chunk_length * self.past_cov_dim
        else:
            past_covariates_flat_dim = 0

        # Future covariates handling
        self.future_cov_projection = None
        if future_cov_dim and self.temporal_width_future:
            self.future_cov_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=temporal_hidden_size_future,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * temporal_width_future
        elif future_cov_dim:
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0

        encoder_dim = (
            self.input_chunk_length * output_dim
            + past_covariates_flat_dim
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim
                * self.output_chunk_length
                * self.nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        decoder_input_dim = decoder_output_dim * self.nr_params
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length * self.nr_params
        )

    @io_processor
    def forward(
        self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TiDE forward pass.

        Parameters
        ----------
        x_in
            Tuple (x_past, x_future, x_static) with shapes
            (batch, time_steps, components).

        Returns
        -------
        torch.Tensor
            Shape (batch, output_chunk_length, output_dim, nr_params).
        """
        x, x_future_covariates, x_static_covariates = x_in
        x_lookback = x[:, :, : self.output_dim]

        # Future covariates projection
        if self.future_cov_dim:
            x_dynamic_future_covariates = torch.cat(
                [
                    x[:, :, None if self.future_cov_dim == 0 else -self.future_cov_dim :],
                    x_future_covariates,
                ],
                dim=1,
            )
            if self.temporal_width_future:
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        # Past covariates projection
        if self.past_cov_dim:
            x_dynamic_past_covariates = x[
                :, :, self.output_dim : self.output_dim + self.past_cov_dim
            ]
            if self.temporal_width_past:
                x_dynamic_past_covariates = self.past_cov_projection(
                    x_dynamic_past_covariates
                )
        else:
            x_dynamic_past_covariates = None

        # Flatten and concatenate encoder inputs
        encoded = [
            x_lookback,
            x_dynamic_past_covariates,
            x_dynamic_future_covariates,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # Encode and decode
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # Temporal decoder with future covariates
        temporal_decoder_input = [
            decoded,
            (
                x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
                if self.future_cov_dim > 0
                else None
            ),
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]
        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # Lookback skip connection
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        y = temporal_decoded + skip.reshape_as(temporal_decoded)
        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return y


class TiDEModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_past: int = 4,
        temporal_width_future: int = 4,
        temporal_hidden_size_past: int = None,
        temporal_hidden_size_future: int = None,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """TiDE model for long-term time-series forecasting.

        TiDE uses MLP-based encoder-decoders without attention, providing
        competitive performance at lower computational cost than Transformers.

        In the SO-SAFED system, this model is used with Reversible Instance
        Normalization (RIN) to forecast hourly PED patient arrivals, enabling
        demand-responsive physician shift optimization.

        Parameters
        ----------
        input_chunk_length
            Number of past time steps as model input.
        output_chunk_length
            Number of time steps predicted at once.
        num_encoder_layers
            Number of residual blocks in the encoder.
        num_decoder_layers
            Number of residual blocks in the decoder.
        decoder_output_dim
            Dimensionality of the decoder output.
        hidden_size
            Width of hidden layers in the residual blocks.
        temporal_width_past
            Width of past covariate projection (0 = raw features).
        temporal_width_future
            Width of future covariate projection (0 = raw features).
        temporal_decoder_hidden
            Width of temporal decoder hidden layers.
        use_layer_norm
            Whether to use layer normalization.
        dropout
            Dropout probability.
        use_static_covariates
            Whether to use static covariates.

        References
        ----------
        .. [1] A. Das et al. "Long-term Forecasting with TiDE: Time-series
               Dense Encoder", http://arxiv.org/abs/2304.08424
        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate
               Time-Series Forecasting against Distribution Shift",
               https://openreview.net/forum?id=cGDAkQo1C0p
        """
        if temporal_width_past < 0 or temporal_width_future < 0:
            raise_log(
                ValueError(
                    "`temporal_width_past` and `temporal_width_future` must be >= 0."
                ),
                logger=logger,
            )
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.temporal_hidden_size_past = temporal_hidden_size_past or hidden_size
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden

        self._considers_static_covariates = use_static_covariates

        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> torch.nn.Module:
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = (
            past_target.shape[1]
            + (past_covariates.shape[1] if past_covariates is not None else 0)
            + (
                historic_future_covariates.shape[1]
                if historic_future_covariates is not None
                else 0
            )
        )

        output_dim = future_target.shape[1]
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        past_cov_dim = input_dim - output_dim - future_cov_dim
        if past_cov_dim and self.temporal_width_past >= past_cov_dim:
            logger.warning(
                f"number of `past_covariates` features is <= `temporal_width_past`, "
                f"leading to feature expansion. "
                f"number of covariates: {past_cov_dim}, "
                f"`temporal_width_past={self.temporal_width_past}`."
            )
        if future_cov_dim and self.temporal_width_future >= future_cov_dim:
            logger.warning(
                f"number of `future_covariates` features is <= `temporal_width_future`, "
                f"leading to feature expansion. "
                f"number of covariates: {future_cov_dim}, "
                f"`temporal_width_future={self.temporal_width_future}`."
            )

        return _TideModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            hidden_size=self.hidden_size,
            temporal_width_past=self.temporal_width_past,
            temporal_width_future=self.temporal_width_future,
            temporal_hidden_size_past=self.temporal_hidden_size_past,
            temporal_hidden_size_future=self.temporal_hidden_size_future,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _check_ckpt_parameters(self, tfm_save):
        new_params = ["temporal_hidden_size_past", "temporal_hidden_size_future"]
        for param in new_params:
            if param not in tfm_save.model_params:
                tfm_save.model_params[param] = None
        super()._check_ckpt_parameters(tfm_save)
