"""
This module contains functions for getting covariate data.
"""

import os.path
from collections import namedtuple

import pandas as pd

from src import paths
from src.trend_forecast.covariate_getters import *

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
        data["max_rel_humidity"] = get_max_rel_humidity(
            loc_code, target_date, series_length
        )
    if covariates.sun_duration:
        data["sun_duration"] = get_sun_duration(loc_code, target_date, series_length)
    if covariates.wind_speed:
        data["wind_speed"] = get_wind_speed(loc_code, target_date, series_length)
    if covariates.radiation:
        data["swave_radiation"] = get_radiation(loc_code, target_date, series_length)
    if covariates.google_search:
        data["google_search"] = get_google_search(loc_code, target_date, series_length)
    if covariates.movement:
        data["movement"] = get_movement_data(loc_code, target_date, series_length)

    return pd.DataFrame.from_dict(data)


def output_covariates_to_csv(
    covariate_data: pd.DataFrame, loc_code: str, target_date: str
) -> str:
    """
    Outputs a covariate dataframe into a CSV file.

    Returns:
        An absolute file path to the covariate CSV file.
    """
    file_dir = os.path.join(paths.OUTPUT_DIR, "covariates", loc_code)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, f"{target_date}.csv")
    covariate_data.to_csv(file_path, index=False)
    return file_path
