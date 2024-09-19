import subprocess
from os import path, makedirs

import jax.numpy as jnp
import pandas as pd
from jax import Array
from numpy import array as np_array
from numpy import savetxt as np_savetxt
from multiprocessing import Pool

from src import paths
from src.trend_forecast.covariates import (
    CovariateSelection,
    get_covariate_data,
    output_covariates_to_csv,
)

selected_covariates = CovariateSelection(
    mean_temp=True,
    max_rel_humidity=True,
    sun_duration=True,
    wind_speed=True,
    radiation=True,
    google_search=False,
    movement=False,
)


def main(beta_estimates: Array, loc_code: str, target_date: str) -> Array:
    """
    Main logic to forecast beta values 28 days into the future.

    Args:
        beta_estimates: Array of estimated beta values.
        loc_code: 2-digit location code.
        target_date: ISO 8601 format. Date we are forecasting from.

    Returns:
        An array of forecasted beta values.
    """
    beta_estimates_path = beta_estimates_to_csv(beta_estimates, loc_code, target_date)

    # Collect and output covariate data for the
    # R subprocess to use.
    covariate_df = get_covariate_data(
        covariates=selected_covariates,
        loc_code=loc_code,
        target_date=target_date,
        series_length=len(beta_estimates),
    )

    covariates_path = output_covariates_to_csv(
        covariate_data=covariate_df, loc_code=loc_code, target_date=target_date
    )

    forecast_file_path = run_r_subprocess(
        loc_code, target_date, beta_estimates_path, covariates_path
    )

    beta_forecast = load_beta_forecast(forecast_file_path)
    return beta_forecast


def beta_estimates_to_csv(
    beta_estimates: Array, loc_code: str, target_date: str
) -> str:
    """
    Saves an array of estimated beta values to a CSV file.
    This allows the beta values to be used by the R script.

    Args:
        beta_estimates: Array of estimated beta values.
        loc_code: 2-digit location code.
        target_date: ISO 8601 format. Date we are forecasting from.

    Returns:
        The absolute file path to the CSV output file.
    """
    output_file_path = path.join(
        paths.PF_OUTPUT_DIR, target_date, f"{loc_code}_beta_estimates.csv"
    )
    numpy_array = np_array(beta_estimates)
    np_savetxt(
        output_file_path, numpy_array, delimiter=",", header="Value", comments=""
    )
    return output_file_path


def run_r_subprocess(
    loc_code: str, target_date: str, beta_estimates_path: str, covariates_path: str
) -> str:
    """
    The current iteration of the R script (as of Sep 7, 2024)
    expects 4 command line args:
        input.betas.path, input.covariates.path,
        func.lib.path, and output.path
    They must be provided in this exact order.

    These absolute paths are set in this function and passed
    into the R subprocess. The R subprocess saves its results
    to a CSV file at output_path.

    Returns:
        The absolute path to the CSV output file, which contains
        the forecasted beta values from the R script.
    """
    input_betas_path = beta_estimates_path
    input_covariates_path = covariates_path
    func_lib_path = path.join(paths.TREND_FORECAST_DIR, "helper_functions.R")
    output_path = path.join(
        paths.TREND_OUTPUT_DIR, target_date, f"{loc_code}_beta_forecast.csv"
    )
    main_script_path = path.join(paths.TREND_FORECAST_DIR, "trend_forecast.R")
    r_working_dir = path.join(paths.TREND_OUTPUT_DIR, target_date)
    cmd = [
        "Rscript",
        main_script_path,
        input_betas_path,
        input_covariates_path,
        func_lib_path,
        output_path,
        r_working_dir,
        target_date,
    ]
    # Ensure the directories exist
    makedirs(path.dirname(output_path), exist_ok=True)
    makedirs(r_working_dir, exist_ok=True)

    # Run the R subprocess and capture the output in real time
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Read the output and error streams as they are generated
    for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line, end="")  # Print R output to Python console
    for stderr_line in iter(process.stderr.readline, ""):
        print(stderr_line, end="")  # Print R errors to Python console

    return output_path


def load_beta_forecast(beta_forecast_file_path: str) -> Array:
    """
    Loads the R-script's Beta forecast.

    The R subprocess saves the generated forecasts to CSV,
    so we need to reload the forecast data back into Python.

    Returns:
        An array of forecasted beta values.
    """
    forecast_df = pd.read_csv(beta_forecast_file_path)
    beta_forecast_array = jnp.asarray(forecast_df.values)
    return beta_forecast_array


def process_dates(dates):
    for date in dates:
        run_r_subprocess(
            loc_code="04",
            target_date=date,
            beta_estimates_path=path.join(paths.PF_OUTPUT_DIR, str(date), "04.csv"),
            covariates_path=path.join(
                paths.OUTPUT_DIR, "covariates", "06", "2023-10-28.csv"
            ),
        )


def parallel_test():
    target_dates = [
        "2023-10-28",
        "2023-11-04",
        "2023-11-11",
        "2023-11-18",
        "2023-11-25",
        "2023-12-02",
        "2023-12-09",
        "2023-12-16",
        "2023-12-23",
        "2023-12-30",
        "2024-01-06",
        "2024-01-13",
        "2024-01-20",
        "2024-01-27",
        "2024-02-03",
        "2024-02-10",
        "2024-02-17",
        "2024-02-24",
        "2024-03-02",
        "2024-03-09",
        "2024-03-16",
        "2024-03-23",
        "2024-03-30",
        "2024-04-06",
    ]

    # Split the 24 dates into 4 chunks of 6 dates each.
    chunks = [target_dates[i : i + 6] for i in range(0, len(target_dates), 6)]

    with Pool() as pool:
        pool.map(process_dates, chunks)


# Used for testing, or manual operation:
if __name__ == "__main__":
    parallel_test()
