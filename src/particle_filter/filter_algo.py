import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from jax import Array
from jax.typing import ArrayLike
from tqdm.notebook import tqdm

import paths
from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.observation_data import ObservationData
from src.particle_filter.output_handler import OutputHandler
from src.particle_filter.particle_cloud import ParticleCloud
from src.particle_filter.transition import OUModel
from logging import Logger

from jax import random
from collections import namedtuple

PFOutput = namedtuple("PFOutput", ["states", "likelihood", "hosp_estimates", "betas"])


class ParticleFilterAlgo:
    def __init__(self, settings: GlobalSettings, logger: Logger) -> None:
        self.settings = settings
        self.likelihoods = []
        self.logger = logger
        self.key = random.PRNGKey(47)

    def run(self, observation_data: ArrayLike, theta: Dict[str, Any]) -> PFOutput:
        """Main logic for running the particle filter.

        Args:
            observation_data: Reported daily hospitalization cases.
                Must be an array of length runtime.
            theta: dictionary {param_name: value} containing parameters proposed
                by the MCMC algorithm.

        Returns:
            PFOutput object. Contains attributes of interest for use in MCMC wrapper.
        """
        config_path = os.path.join(paths.PF_DIR, "config.toml")

        particles = ParticleCloud(
            settings=self.settings,
            transition=OUModel(config_file=config_path),
            logger=self.logger,
            theta=theta,
        )

        # Initialize an object that stores the hospitalization data.
        if len(observation_data) != self.settings.runtime:
            raise AssertionError(
                "The length of observation_data must be equal to runtime."
            )
        observed_data = ObservationData(observation_data)

        # tqdm provides the console progress bar.
        for t in tqdm(
            range(self.settings.runtime),
            desc="Running Particle Filter",
            colour="green",
            leave=False,
        ):
            self.key, *subkeys = random.split(self.key)
            if t != 0:
                # If t = 0, then we just initialized the particles. Thus, no update.
                particles.update_all_particles(t=t)

            case_report = observed_data.get_observation(t=t)

            particles.compute_all_weights(reported_data=case_report, t=t)
            particles.normalize_weights(t=t)
            particles.resample(t=t)
            particles.perturb_beta(t=t)

        output_object = PFOutput(
            states=particles.states,
            likelihood=particles.likelihoods,
            hosp_estimates=particles.hosp_estimates,
            betas=particles.betas,
        )

        return output_object
