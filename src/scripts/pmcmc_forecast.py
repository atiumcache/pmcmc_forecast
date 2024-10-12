"""
This script orchestrates the logic for
running the forecasting pipeline on all locations
over a range of specified forecasting dates.

This is used for generating forecasts on past data
(for Fall 2023 through Spring 2024),
so we can compare the performance of the PMCMC forecast
pipeline to other models that we have used in the past.
"""

from jax import Array

from hosp_forecast import main as hosp_forecast
from pmcmc import main as pmcmc
from trend_forecast import main as trend_forecast


def main():
    pass


def single_location_pipeline(loc_code: str, target_date: str) -> None:
    estimated_betas = pmcmc.main(location_code=loc_code, target_date=target_date)
    forecasted_betas = trend_forecast.main(estimated_betas, loc_code, target_date)
    # TODO: hosp_forecast needs to return array of forecasts (uncertainty)
    hosp_forecast.main(forecasted_betas, loc_code, target_date)


def output_forecast_to_csv(
    loc_code: str, target_date: str, hosp_forecasts: Array
) -> None:
    """
    Outputs the hospitalization forecasts to a csv file.

    Args:
        loc_code: 2-digit location code.
        target_date: ISO8601 format. Date we are forecasting from.
        hosp_forecasts: An array of hospitalization forecasts.

    Returns
    """
    raise NotImplementedError("TODO")
