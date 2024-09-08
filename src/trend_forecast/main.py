import subprocess
from os import path

import jax.numpy as jnp
import pandas as pd
from jax import Array
from numpy import array as np_array
from numpy import savetxt as np_savetxt

from src import paths


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
    forecast_file_path = run_r_subprocess(loc_code, target_date, beta_estimates_path)
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


def run_r_subprocess(loc_code: str, target_date: str, beta_estimates_path: str) -> str:
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
    input_covariates_path = ""
    func_lib_path = path.join(paths.TREND_FORECAST_DIR, "helper_functions.R")
    output_path = path.join(
        paths.TREND_OUTPUT_DIR, target_date, f"{loc_code}_beta_forecast.csv"
    )
    main_script_path = path.join(paths.TREND_FORECAST_DIR, "trend_forecast.R")

    subprocess.run(
        [
            "Rscript",
            main_script_path,
            input_betas_path,
            input_covariates_path,
            func_lib_path,
            output_path,
        ],
        capture_output=True,
        text=True,
    )
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
