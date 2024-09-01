"""
This is the main script to run the PMCMC algorithm
on a single date and location. 

Outputs estimated beta values that can then be processed
by the Trend Forecasting R script. 
"""

from os import path
from typing import Callable

import pandas as pd
import toml
from jax import Array
from jax.typing import ArrayLike

from src import paths
from src.pmcmc.helpers import get_data_since_week_26
from src.pmcmc.location import Location
from src.pmcmc.pmcmc import PMCMC
from src.pmcmc.prior import UniformPrior, Prior


def main(location_code: str, target_date: str) -> Array:
    """
    Runs a PMCMC algorithm on a single date and location.
    Gathers location-level data hospitalization data.
    Then,

    Args:
        location_code: A 2-digit code corresponding to a location. See `/datasets/locations.csv`.
        target_date: The date we will forecast from.

    Returns:
          An array of estimated beta values for each step in the time series.
    """
    observations, location = get_hosp_observations(
        location_code, target_date
    )

    # Determine number of days for PF to estimate, based on length of data.
    time_steps = len(observations)

    location_info = {
        "population": location.population,
        "location_code": location_code,
        "target_date": target_date,
        "runtime": time_steps,
    }

    config_path = path.join(paths.PMCMC_DIR, "config.toml")
    config = toml.load(config_path)

    prior = UniformPrior()

    pmcmc_algo = PMCMC(
        iterations=config["mcmc"]["iterations"],
        init_thetas=config["mcmc"]["initial_theta"],
        prior=prior,
        location_info=location_info,
        observation_data=observations,
        burn_in=config["mcmc"]["burn_in"],
    )
    pmcmc_algo.run()

    # TODO: Output betas (and other data?) to csv or database for analysis.

    return pmcmc_algo.mle_betas


def get_hosp_observations(location_code, target_date) -> (ArrayLike, Location):
    """
    Returns an array of hospital observations.
    Time series starts on 2023-06-25 and ends on target_date.
    """
    location = Location(location_code)
    target_date = pd.to_datetime(target_date)
    filtered_state_data = get_data_since_week_26(location.hosp_data, target_date)
    observations = filtered_state_data[
        "previous_day_admission_influenza_confirmed"
    ].values
    return observations, location


def get_prior_function() -> Callable | Prior:
    return UniformPrior()
