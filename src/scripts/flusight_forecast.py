"""
This script orchestrates running a PMCMC forecast
algorithm for the 2024 FluSight forecasting challenge.

The pipeline is run once for each location.
"""
import sys
from jax import Array

from src.hosp_forecast import main as hosp_forecast
from src.pmcmc import main as pmcmc
from src.trend_forecast import main as trend_forecast


def single_location_pipeline(loc_code: str, target_date: str) -> None:
    """
    Forecast a single location.

    Args:
        loc_code: 2-digit string. See datasets/locations.csv
        target_date: the date we are forecasting from.

    Returns:
        None. Hosp_forecast outputs the forecast to csv.
    """
    estimated_betas = pmcmc.main(location_code=loc_code, target_date=target_date)
    forecasted_betas = trend_forecast.main(estimated_betas, loc_code, target_date)
    # Currently, hosp_forecast.main outputs to a csv file, so nothing is returned.
    hosp_forecast.main(forecasted_betas, loc_code, target_date)


def main():
    raise NotImplementedError("This method has not yet been implemented.")
