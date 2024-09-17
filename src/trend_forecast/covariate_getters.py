"""
Contains functions for collecting various covariate data,
that will then be used in the Trend Forecasting algorithm.
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


def get_daily_weather_data(loc_code: str, target_date: str, variable: str, series_length: int) -> Series:
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
    data_type = 'daily'
    hourly_variables = ['relative_humidity_2m']
    if variable in hourly_variables:
        data_type = 'hourly'

    # Get latitude and longitude from loc_code
    latitude, longitude = get_lat_long(loc_code)

    # Convert target_date string to datetime object
    target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")

    # Calculate the start date (80 days before target_date)
    start_date_dt = target_date_dt - timedelta(days=(series_length - 1))

    # Format dates as strings in ISO 8601 format
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = target_date_dt.strftime("%Y-%m-%d")

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
    if data_type == 'hourly':
        times = data['hourly']['time']
        values = data['hourly'][variable]
        
        df = pd.DataFrame({
            'time': pd.to_datetime(times),
            variable: values
        })
        
        df.set_index('time', inplace=True)
        
        # Resample the data to daily frequency and take the max for each day
        daily_max = df.resample('D').max()
        return daily_max[variable]

    else:  # data is provided in daily format
        dates = data["daily"]["time"]
        values = data["daily"].get(variable, [])
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
    return get_daily_weather_data(loc_code, target_date, "temperature_2m_mean", series_length)


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
    return get_daily_weather_data(loc_code, target_date, "relative_humidity_2m", series_length)


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
    return get_daily_weather_data(loc_code, target_date, 'sunshine_duration', series_length)


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
    return get_daily_weather_data(loc_code, target_date, 'wind_speed_10m_max', series_length)


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
    return get_daily_weather_data(loc_code, target_date, 'wind_speed_10m_max', series_length)


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
