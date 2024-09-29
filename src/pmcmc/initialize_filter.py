import os
from logging import Logger

import toml

from src import paths
from src.pmcmc.filter_algo import ParticleFilterAlgo
from src.pmcmc.global_settings import GlobalSettings


def initialize_particle_filter(
    state_population: int,
    location_code: str,
    target_date: str,
    runtime: int,
    logger: Logger,
    dispersion: float
) -> ParticleFilterAlgo:
    """Initializes a ParticleFilterAlgo object."""

    config = load_config()

    global_settings = GlobalSettings(
        num_particles=config["filter_params"]["num_particles"],
        population=state_population,
        location_code=location_code,
        final_date=target_date,
        runtime=runtime,
        dt=config["filter_params"]["dt"],
        beta_prior=tuple(config["filter_params"]["beta_prior"]),
        seed_size=config["filter_params"]["seed_size"],
        dispersion=dispersion,
    )

    pf_algo = ParticleFilterAlgo(settings=global_settings, logger=logger)
    return pf_algo


def load_config():
    config_path = os.path.join(paths.PMCMC_DIR, "config.toml")
    return toml.load(config_path)
