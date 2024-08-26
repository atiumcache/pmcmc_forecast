"""
This is the main script to run the PMCMC algorithm
on a single date and location. 

Outputs estimated beta values that can then be processed
by the Trend Forecasting R script. 
"""

from os import path

import pandas as pd
import toml
from jax import Array

from src import paths
from src.pmcmc.helpers import get_data_since_week_26
from src.pmcmc.location import Location
from src.pmcmc.pmcmc import PMCMC
from src.pmcmc.prior import UniformPrior


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
    location = Location(location_code)

    target_date = pd.to_datetime(target_date)

    filtered_state_data = get_data_since_week_26(location.hosp_data, target_date)
    observations = filtered_state_data[
        "previous_day_admission_influenza_confirmed"
    ].values

    # Determine number of days for PF to estimate, based on length of data.
    time_steps = len(observations)

    location_info = {
        "population": location.population,
        "location_code": location_code,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "runtime": time_steps,
    }

    config_path = path.join(paths.PMCMC_DIR, "config.toml")
    config = toml.load(config_path)

    prior = UniformPrior()

    pmcmc_algo = PMCMC(
        iterations=config["mcmc"]["iterations"],
        init_theta=config["mcmc"]["initial_theta"],
        prior=prior,
        location_info=location_info,
        observation_data=observations,
    )
    pmcmc_algo.run()

    # TODO: Output betas (and other data?) to csv or database for record keeping, and for use by Trend Forecasting.

    return pmcmc_algo.mle_betas
