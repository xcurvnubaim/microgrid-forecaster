"""Build the Chronos-2 foundation forecaster (notebook's final model choice).

Foundation models are zero-shot: ``fit`` only stores the historical context,
no weights are learned. The heavy backend (``chronos-forecasting`` + torch) is
imported lazily by skforecast's adapter, so importing this module stays cheap.
"""

from __future__ import annotations

from skforecast.foundation import ForecasterFoundation, FoundationModel

from ..config import ModelCfg


def build_forecaster(cfg: ModelCfg) -> ForecasterFoundation:
    estimator = FoundationModel(
        model_id=cfg.model_id,
        context_length=cfg.context_length,
    )
    return ForecasterFoundation(estimator=estimator)
