"""
Contains functions for getting various covariate data,
that will then be used in the Trend Forecasting script.
"""

from datetime import datetime, timedelta
from os import path
from typing import Tuple

import pandas as pd
import requests
from pandas import Series

from src import paths


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


def get_daily_weather_data(loc_code: str, target_date: str, variable: str) -> Series:
    """
    Retrieves daily weather data for a specific location from 80 days before the target date up to the target date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        variable: The weather variable to retrieve (e.g., 'temperature_2m_mean').

    Returns:
        pd.Series: A time series of the requested weather variable for the last 80 days up to the target date.
    """
    # Get latitude and longitude from loc_code
    latitude, longitude = get_lat_long(loc_code)

    # Convert target_date string to datetime object
    target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")

    # Calculate the start date (80 days before target_date)
    start_date_dt = target_date_dt - timedelta(days=80)

    # Format dates as strings in ISO 8601 format
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = target_date_dt.strftime("%Y-%m-%d")

    # Open Meteo API request URL for the specified weather variable
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={variable}"
        f"&timezone=auto"
    )

    # Send the GET request to Open Meteo API
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    # Parse the JSON response
    data = response.json()

    # Extract the data for the requested variable
    dates = data["daily"]["time"]
    values = data["daily"].get(variable, [])

    # Create a Pandas Series with the date as the index and the weather variable value as the value
    weather_series = pd.Series(data=values, index=pd.to_datetime(dates), name=variable)

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
    return get_daily_weather_data(loc_code, target_date, "temperature_2m_mean")


def get_max_rel_humidity(loc_code: str, target_date: str, series_length: int) -> list:
    """
    Retrieves the maximum relative humidity for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of maximum relative humidity values for the given location and date.
    """
    raise NotImplementedError("get_max_rel_humidity is not yet implemented.")


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
    return get_daily_weather_data(loc_code, target_date, 'sunshine_duration')


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
    return get_daily_weather_data(loc_code, target_date, 'wind_speed_10m_max')


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
    return get_daily_weather_data(loc_code, target_date, 'wind_speed_10m_max')


def get_google_search(loc_code: str, target_date: str, series_length: int) -> list:
    """
    Retrieves Google search trend data for a specific location and date.

    Args:
        loc_code: A 2-digit string representing the location code.
        target_date: An ISO 8601 formatted date string (YYYY-MM-DD).
        series_length: Number of days prior to target date.

    Returns:
        A list of Google search trend scores for the given location and date.
    """
    raise NotImplementedError("get_google_search is not yet implemented.")


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
