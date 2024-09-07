import subprocess
from os import path
from src import paths

from jax import Array


def main(beta_estimates: Array, loc_code: str, target_date: str) -> Array:
    """
    Main logic to forecast beta values 28 days into the future.

    Args:
        beta_estimates: Array of estimated beta values.
        loc_code: 2-digit location code.
        target_date: ISO 8601 format. Date we are forecasting from.

    Returns:
        beta_forecast: An array of forecasted beta values.
    """
    beta_estimates_path = beta_estimates_to_csv(beta_estimates, loc_code, target_date)
    run_r_subprocess(loc_code, target_date, beta_estimates_path)
    beta_forecast = load_beta_forecast(loc_code, target_date)
    return beta_forecast


def beta_estimates_to_csv(
    beta_estimates: Array, loc_code: str, target_date: str
) -> str:
    pass


def run_r_subprocess(loc_code: str, target_date: str, beta_estimates_path: str) -> None:
    """
    The current iteration of the R script (as of Sep 7, 2024)
    expects 4 command line args:
        input.betas.path, input.covariates.path,
        func.lib.path, and output.path

    These absolute paths are set in this function and passed
    into the R subprocess.

    Returns:
        None. The R subprocess saves the output directly
        to the output.path csv file.
    """
    input_betas_path = beta_estimates_path
    input_covariates_path = ""
    func_lib_path = path.join(paths.TREND_FORECAST_DIR, 'helper_functions.R')
    output_path = ""
    main_file_path = path.join(paths.TREND_FORECAST_DIR, 'trend_forecast.R')

    subprocess.run(
        [
            "Rscript",
            main_file_path,
            input_betas_path,
            input_covariates_path,
            func_lib_path,
            output_path,
        ],
        capture_output=True,
        text=True,
    )


def load_beta_forecast(loc_code: str, target_date: str) -> Array:
    """
    Loads the R-script's Beta forecast.

    The R subprocess saves the generated forecasts to CSV,
    so we need to reload the forecast data back into Python.

    Returns:
        An array of forecasted beta values.
    """
    pass
