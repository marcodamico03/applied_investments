"""
Asset growth signal (Cooper, Gulen and Schill 2008).
Growth of total assets over the last fiscal year: at(t) / at(t-1) - 1.
"""
import pandas as pd

# Compustat columns this signal needs (passed to load_panel)
COMPUSTAT_COLS = ["at", "at_lag"]
REBALANCE = "annual"


def asset_growth(panel: pd.DataFrame) -> pd.Series:
    return panel["at"] / panel["at_lag"] - 1