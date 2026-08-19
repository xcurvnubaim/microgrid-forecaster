"""microgrid_forecaster — configurable PV and demand forecaster (Chronos-2).

Forecasts pv_avg and demand at hourly resolution, using ECMWF weather as
exogenous features. Chronos-2 is zero-shot (pre-trained, no learning on our
data). Two modes: `evaluate` holds out historical CSV and reports accuracy;
`production` feeds fresh history + weather and serves into Redis + HTTP.
"""

__version__ = "0.1.0"

SERIES = ("pv_avg", "demand")
DEFAULT_HORIZON_H = 24
