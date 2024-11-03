"""
Contains functions for collecting various covariate data,
that will then be used in the Trend Forecasting algorithm.
"""

from datetime import datetime, timedelta
from os import path
from typing import Tuple

import pandas as pd
import requests
from numpy import ndarray
from pandas import Series
from pytrends.request import TrendReq

from src.utils import paths


def get_lat_long(loc_code: str) -> Tuple[float, float]:
    """
    Returns a tuple of latitude and longitude coordinates
    for the specified location.
    """
    loc_code = loc_code.zfill(2)
    csv_path = path.join(paths.DATASETS_DIR, "locations_with_capitals.csv")
    df = pd.read_csv(csv_path)
    lat = df["latitude"].loc[df["location"] == loc_code].values[0]
    long = df["longitude"].loc[df["location"] == loc_code].values[0]
    return lat, long


def get_start_end_dates(
    target_date: str, series_length: int, lag: int = 0
) -> Tuple[str, str]:
    """
    Calculates start and end dates for a time series.

    Args:
        target_date: The day we will forecast from. If lag == 0, this
            is the final day of the time series.
        series_length: number of days in the time series.
        lag: Number of days to shift the time series by. A positive number x
            will shift the time series x days into the past. Default is 0.

    Returns:
         A tuple containing the start and end dates for the specified series.
    """
    target_date_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=lag)

    # Calculate the start date
    start_date_dt = target_date_dt - timedelta(days=(series_length - 1))

    # Format dates as strings in ISO 8601 format
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = target_date_dt.strftime("%Y-%m-%d")
    return start_date, end_date


def get_daily_weather_data(
    loc_code: str, target_date: str, variable: str, series_length: int
) -> Series:
    """
    Retrieves daily weather data for a specific location from 80 days before the target date up to the target date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        variable: The weather variable to retrieve (e.g., 'temperature_2m_mean').
        series_length: The length of the series to retrieve.

    Returns:
        pd.Series: A time series of the requested weather variable for the last 80 days up to the target date.
    """
    data_type = "daily"
    hourly_variables = ["relative_humidity_2m"]
    if variable in hourly_variables:
        data_type = "hourly"

    # Get latitude and longitude from loc_code
    latitude, longitude = get_lat_long(loc_code)

    start_date, end_date = get_start_end_dates(target_date, series_length)

    # Open Meteo API request URL for the specified weather variable
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&{data_type}={variable}"
        f"&timezone=auto"
    )

    # Send the GET request to Open Meteo API
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.json()

    # Some weather data is only provided in hourly format,
    # so we need to process it further to convert to daily.
    if data_type == "hourly":
        times = data["hourly"]["time"]
        values = data["hourly"][variable]

        df = pd.DataFrame({"time": pd.to_datetime(times), variable: values})

        df.set_index("time", inplace=True)

        # Resample the data to daily frequency and take the max for each day
        daily_max = df.resample("D").max()
        return daily_max[variable]

    else:  # data is provided in daily format
        dates = data["daily"]["time"]
        values = data["daily"].get(variable, [])
        weather_series = pd.Series(
            data=values, index=pd.to_datetime(dates), name=variable
        )
        return weather_series


def get_mean_temp(loc_code: str, target_date: str, series_length: int) -> Series:
    """
    Retrieves the mean temperature for a specific location from 80 days before the target date up to the target date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Length of time series, with target_date as final row.

    Returns:
        A time series of mean temperatures for the last 80 days up to the target date.
    """
    return get_daily_weather_data(
        loc_code, target_date, "temperature_2m_mean", series_length
    )


def get_max_rel_humidity(loc_code: str, target_date: str, series_length: int) -> Series:
    """
    Retrieves the maximum relative humidity for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of maximum relative humidity values for the given location and date.
    """
    return get_daily_weather_data(
        loc_code, target_date, "relative_humidity_2m", series_length
    )


def get_sun_duration(loc_code: str, target_date: str, series_length: int) -> Series:
    """
    Retrieves the duration of sunshine for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of sunshine duration values (in hours) for the given location and date.
    """
    return get_daily_weather_data(
        loc_code, target_date, "sunshine_duration", series_length
    )


def get_wind_speed(loc_code: str, target_date: str, series_length: int) -> Series:
    """
    Retrieves the wind speed for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of wind speed values (in meters per second) for the given location and date.
    """
    return get_daily_weather_data(
        loc_code, target_date, "wind_speed_10m_max", series_length
    )


def get_radiation(loc_code: str, target_date: str, series_length: int) -> Series:
    """
    Retrieves the solar radiation data for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of solar radiation values for the given location and date.
    """
    return get_daily_weather_data(
        loc_code, target_date, "shortwave_radiation_sum", series_length
    )


def convert_loc_code_to_abbrev(loc_code: str) -> str:
    """
    Given a 2-digit location code, returns the 2-letter abbreviation.
    i.e. for Arizona, we have '04' -> 'AZ'
    """
    locations_csv_path = path.join(paths.DATASETS_DIR, "locations.csv")
    locations_df = pd.read_csv(locations_csv_path)

    # Use bitwise OR (|) for multiple conditions and .astype(str) to match types
    matching_rows = locations_df.loc[
        (locations_df["location"] == loc_code)
        | (locations_df["location"] == int(loc_code)),
        "abbreviation",
    ]

    if matching_rows.empty:
        raise ValueError(f"Location code '{loc_code}' not found in locations.csv")

    return matching_rows.values[0]


def get_flusight_google_search(
    loc_code: str, search_term: str, target_date: str, start_date: str
) -> ndarray:
    """
    A special case of get_google_search() for FluSight 2024. We want to
    be able to indicate a start_date for the time series,
    rather than pass in a series length.

    Returns a numpy array time series, from start_date to target_date.
    """
    length = (pd.to_datetime(target_date) - pd.to_datetime(start_date)).days
    df = get_google_search(loc_code, search_term, target_date, length)
    return df.to_numpy().ravel()


def get_google_search(
    loc_code: str, search_term: str, target_date: str, series_length: int
) -> pd.DataFrame:
    """
    Retrieves Google search trend data for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        search_term: Search term to get trend data for.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A time series (list) of Google search trend scores for the given location.
        Time series ends at target_date, and begins series_length days earlier.
    """
    pytrends = TrendReq(hl="en-US", tz=360)

    loc_abbrev = convert_loc_code_to_abbrev(loc_code)

    start_date, end_date = get_start_end_dates(
        target_date=target_date, series_length=series_length, lag=3
    )

    # Set Google geo code. Puerto Rico does not have US prefix.
    geo_code = f"US-{loc_abbrev}"
    if loc_code in [72, "72"]:
        geo_code = "PR"

    kw_list = [search_term]
    pytrends.build_payload(
        kw_list,
        cat=0,
        timeframe=f"{start_date} {end_date}",
        geo=geo_code,
        gprop="",
    )

    data = pytrends.interest_over_time()

    if not data.empty:
        data = data.drop(columns=["isPartial"])
        return data
    else:
        raise ValueError(
            f"Google Trends data failed for {loc_code} from {start_date} to {end_date}."
        )


def get_movement_data(loc_code: str, target_date: str, series_length: int) -> list:
    """
    Retrieves movement data for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of movement data values (e.g., mobility or transport metrics) for the given location and date.
    """
    raise NotImplementedError("get_movement_data is not yet implemented.")
