"""
This module contains functions for getting covariate data.
"""

from collections import namedtuple

import pandas as pd
from covariate_getters import *

CovariateSelection = namedtuple(
    "CovariateSelection",
    [
        "mean_temp",
        "max_rel_humidity",
        "sun_duration",
        "wind_speed",
        "radiation",
        "google_search",
        "movement",
    ],
)


def get_covariate_data(
    covariates: CovariateSelection, loc_code: str, target_date: str, series_length: int
) -> pd.DataFrame:
    """
    Collects covariate data from various sources,
    to be used by Trend Forecasting algorithm.

    Args:
        covariates: A namedtuple containing boolean values for covariate selection.
        loc_code: A 2-digit string corresponding to a location.
        target_date: ISO 8601 date string. The date we are forecasting from.
        series_length: The length of the covariate time series to get.
                       Final date of series is the target_date.

    Returns:
        A dataframe with the selected covariate data.
    """
    data = {}

    if covariates.mean_temp:
        data["mean_temp"] = get_mean_temp(loc_code, target_date, series_length)
    if covariates.max_rel_humidity:
        data["max_rel_humidity"] = get_max_rel_humidity(loc_code, target_date, series_length)
    if covariates.sun_duration:
        data["sun_duration"] = get_sun_duration(loc_code, target_date, series_length)
    if covariates.wind_speed:
        data["wind_speed"] = get_wind_speed(loc_code, target_date, series_length)
    if covariates.radiation:
        data["radiation"] = get_radiation(loc_code, target_date, series_length)
    if covariates.google_search:
        data["google_search"] = get_google_search(loc_code, target_date, series_length)
    if covariates.movement:
        data["movement"] = get_movement_data(loc_code, target_date, series_length)

    return pd.DataFrame.from_dict(data)


def output_covariates_to_csv(covariate_data: pd.DataFrame,
                             loc_code: str,
                             target_date: str) -> None:
    pass
