import logging
import os
from datetime import datetime, timedelta
from typing import NamedTuple

import pandas as pd
import sklearn.preprocessing as sp
from tqdm.notebook import tqdm

from src.trend_forecast.covariate_getters import get_flusight_google_search
from src.trend_forecast.covariates import get_covariate_data
from src.utils import paths


class CovariateSelection(NamedTuple):
    mean_temp: bool
    max_rel_humidity: bool
    sun_duration: bool
    wind_speed: bool
    radiation: bool
    google_search: bool
    movement: bool


def setup_covariate_logger() -> logging.Logger:
    """
    Sets up a logger that writes to a new file each time.
    """
    log_dir = os.path.join(paths.DATASETS_DIR, "covariates", "logs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(
        log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def generate_all_covariate_csv_files(date: str, series_length: int):
    """
    Generates all covariate csv files for a given date and series length.
    """
    covariate_database_dir = os.path.join(paths.DATASETS_DIR, "covariates", "database")

    for file_path in tqdm(os.listdir(covariate_database_dir)):
        full_file_path = os.path.join(covariate_database_dir, file_path)
        generate_single_covariate_csv_files(
            date=date, series_length=series_length, file_path=full_file_path
        )


def generate_single_covariate_csv_files(date: str, series_length: int, file_path: str):
    """
    Generates a covariate csv file for a given date and series length.

    These files are specifically formulated to be fed into
    the Trend Forecasting algorithm. Implements scaling for the features.

    Args:
        date: The date we will be forecasting from.
        series_length: The length of the series.
        file_path: The path to the database csv file.

    Returns:
        None. Files are output to /datasets/covariates/date/{loc_code}.csv
    """
    # Get location code from file path
    loc_code = os.path.basename(file_path)[
        :2
    ]  # Assuming filename starts with location code

    df = pd.read_csv(file_path)

    if pd.to_datetime(df['date'].iloc[-1]) < pd.to_datetime(date):
        raise ValueError(f"Covariate database needs to be updated to at least {date}.")

    subset_df = get_subset_df(df, date, series_length)

    # Scale the data
    scaler = sp.StandardScaler()
    features_df = subset_df.drop(columns=['date'])
    scaled_features = scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
    scaled_df.insert(0, 'time_0', list(range(len(scaled_df))))
    
    # Output to csv
    output_dir = os.path.join(paths.DATASETS_DIR, "covariates", date)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{loc_code}.csv")
    scaled_df.to_csv(output_path, index=False)


def get_subset_df(df: pd.DataFrame, final_date: str, length: int):
    """
    Returns a subset of the data based on the final_date and series length.
    """
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate the start date for the subset
    end_date = pd.to_datetime(final_date)
    start_date = end_date - pd.Timedelta(days=length - 1)
    
    # Filter the DataFrame to get the desired subset
    subset_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    return subset_df


def update_all_covariate_data():
    """
    Updates all covariate data for all locations.
    """
    logger = setup_covariate_logger()
    location_csv_path = os.path.join(paths.DATASETS_DIR, "locations.csv")
    location_df = pd.read_csv(location_csv_path)
    location_codes = location_df["location"]

    for loc_code in tqdm(
        location_codes, desc="Updating covariate data", colour="purple"
    ):
        loc_file_path = os.path.join(
            paths.DATASETS_DIR,
            "covariates",
            "database",
            f"{str(loc_code).zfill(2)}.csv",
        )
        update_covariate_data(file_path=loc_file_path, logger=logger)


def update_covariate_data(file_path: str, logger: logging.Logger):
    """
    Updates a covariate file up to the current date for one location.

    Args:
        file_path: the absolute file path to a single location's covariate csv.
        logger: the logger to use.

    Returns:
        None. Updates the csv files located in /datasets/covariates/.
    """

    # Read existing data
    old_df = pd.read_csv(file_path)
    old_df["date"] = pd.to_datetime(old_df["date"])
    old_df.drop(columns=["google_search"], inplace=True)
    last_date = old_df.iloc[-1]["date"]

    # Get location code from file path
    loc_code = os.path.basename(file_path)[
        :2
    ]  # Assuming filename starts with location code

    # Calculate dates needed
    current_date = datetime.now().strftime("%Y-%m-%d")
    days_needed = (pd.to_datetime(current_date) - last_date).days

    if days_needed <= 0:
        print(f"Data is already up to date for location {loc_code}")
        return

    # Determine which covariates are present in the existing file
    covariates = CovariateSelection(
        mean_temp="mean_temp" in old_df.columns,
        max_rel_humidity="max_rel_humidity" in old_df.columns,
        sun_duration="sun_duration" in old_df.columns,
        wind_speed="wind_speed" in old_df.columns,
        radiation="swave_radiation" in old_df.columns,
        google_search=False,  # Google Search needs separate function (below)
        movement="movement" in old_df.columns,
    )

    try:
        # Get new data
        new_data = get_covariate_data(
            covariates=covariates,
            loc_code=loc_code,
            target_date=current_date,
            series_length=days_needed,
        )

        # Add date column to new data
        new_data["date"] = pd.date_range(
            start=last_date + timedelta(days=1),
            end=pd.to_datetime(current_date),
            freq="D",
        )

        # Combine old and new data
        updated_df = pd.concat([old_df, new_data], ignore_index=True)

        # Ensure no duplicate dates
        updated_df = updated_df.drop_duplicates(subset=["date"], keep="last")

        # Sort by date
        updated_df = updated_df.sort_values("date")

        # Getting Google Search data requires much different
        # logic than the weather data.

        updated_df["google_search"] = get_flusight_google_search(
            loc_code=loc_code,
            search_term="flu symptoms",
            target_date=current_date,
            start_date="2024-08-19",
        )

        # Save updated data
        updated_df.to_csv(file_path, index=False)
        logger.info(f"Successfully updated covariate data for location {loc_code}")

    except Exception as e:
        logger.error(f"Error updating covariate data for location {loc_code}: {str(e)}")
        raise
