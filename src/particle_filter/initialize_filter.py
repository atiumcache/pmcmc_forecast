from logging import Logger

import os

import toml

import paths
from src.particle_filter.filter_algo import ParticleFilterAlgo
from src.particle_filter.global_settings import GlobalSettings


def initialize_particle_filter(
    state_population: int,
    location_code: str,
    target_date: str,
    runtime: int,
    logger: Logger,
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
        dispersion=config["filter_params"]["dispersion"],
    )

    pf_algo = ParticleFilterAlgo(settings=global_settings, logger=logger)
    return pf_algo


def load_config():
    config_path = os.path.join(paths.PF_DIR, "config.toml")
    return toml.load(config_path)
