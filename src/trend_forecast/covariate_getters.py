"""
Contains functions for getting various covariate data,
that will then be used in the Trend Forecasting script.
"""
from typing import Tuple

import pandas as pd
from datetime import datetime, timedelta
import requests

from src import paths
from os import path


def get_lat_long(loc_code: str) -> Tuple[float, float]:
    """
    Returns a tuple of latitude and longitude coordinates
    for the specified location.
    """
    loc_code = loc_code.zfill(2)
    csv_path = path.join(paths.DATASETS_DIR, 'locations_with_capitals.csv')
    df = pd.read_csv(csv_path)
    lat = df["latitude"].loc[df["location"] == loc_code].values[0]
    long = df["longitude"].loc[df["location"] == loc_code].values[0]
    return lat, long


def get_mean_temp(loc_code: str, target_date: str) -> pd.Series:
    """
    Retrieves the mean temperature for a specific location from 80 days before the target date up to the target date.

    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).

    Returns:
        pd.Series: A time series of mean temperatures for the last 80 days up to the target date.
    """
    latitude, longitude = get_lat_long(loc_code)
    target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_date_dt = target_date_dt - timedelta(days=80)

    # Format dates as strings in ISO 8601 format
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = target_date_dt.strftime("%Y-%m-%d")

    print(latitude)
    # Open Meteo API request URL for temperature data
    api_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean"
        f"&timezone=auto"
    )

    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.json()

    dates = data['daily']['time']
    temp_mean = data['daily']['temperature_2m_mean']

    mean_temp_series = pd.Series(data=temp_mean, index=pd.to_datetime(dates), name="mean_temp")

    return mean_temp_series


def get_max_rel_humidity(loc_code: str, target_date: str) -> list:
    """
    Retrieves the maximum relative humidity for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of maximum relative humidity values for the given location and date.
    """
    raise NotImplementedError("get_max_rel_humidity is not yet implemented.")


def get_sun_duration(loc_code: str, target_date: str) -> list:
    """
    Retrieves the duration of sunshine for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of sunshine duration values (in hours) for the given location and date.
    """
    raise NotImplementedError("get_sun_duration is not yet implemented.")


def get_wind_speed(loc_code: str, target_date: str) -> list:
    """
    Retrieves the wind speed for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of wind speed values (in meters per second) for the given location and date.
    """
    raise NotImplementedError("get_wind_speed is not yet implemented.")


def get_radiation(loc_code: str, target_date: str) -> list:
    """
    Retrieves the solar radiation data for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of solar radiation values (in watts per square meter) for the given location and date.
    """
    raise NotImplementedError("get_radiation is not yet implemented.")


def get_google_search(loc_code: str, target_date: str) -> list:
    """
    Retrieves Google search trend data for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of Google search trend scores for the given location and date.
    """
    raise NotImplementedError("get_google_search is not yet implemented.")


def get_movement_data(loc_code: str, target_date: str) -> list:
    """
    Retrieves movement data for a specific location and date.
    
    Args:
        loc_code (str): A 2-digit string representing the location code.
        target_date (str): An ISO 8601 formatted date string (YYYY-MM-DD).
    
    Returns:
        list: A list of movement data values (e.g., mobility or transport metrics) for the given location and date.
    """
    raise NotImplementedError("get_movement_data is not yet implemented.")
