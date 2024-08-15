import os
from typing import Any, Dict, Tuple

from jax import Array
from jax.typing import ArrayLike
from tqdm import tqdm

import paths
from src.particle_filter.global_settings import GlobalSettings
from src.particle_filter.logger import get_logger
from src.particle_filter.observation_data import ObservationData
from src.particle_filter.output_handler import OutputHandler
from src.particle_filter.particle_cloud import ParticleCloud
from src.particle_filter.transition import OUModel


class ParticleFilterAlgo:
    def __init__(self, settings: GlobalSettings, logger) -> None:
        self.settings = settings
        self.likelihoods = []
        self.logger = logger

    def run(
        self, observation_data: ArrayLike, theta: Dict[str, Any]
    ) -> Tuple[Array, Array, Array, Array]:
        """Main logic for running the particle filter.

        Args:
            observation_data: Reported daily hospitalization cases.
                Must be an array of length runtime.
            theta: dictionary {param_name: value} containing parameters proposed
                by the MCMC algorithm.

        Returns:

        """
        config_path = os.path.join(paths.PF_DIR, "config.toml")
        logger = get_logger()

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
            range(self.settings.runtime), desc="Running Particle Filter", colour="green"
        ):
            if t != 0:
                # If t = 0, then we just initialized the particles. Thus, no update.
                particles.update_all_particles(t)

            case_report = observed_data.get_observation(t)

            particles.compute_all_weights(reported_data=case_report, t=t)
            particles.normalize_weights(t=t)
            particles.resample(t=t)
            particles.perturb_beta(t=t)

        return (
            particles.likelihoods,
            particles.hosp_estimates,
            particles.states,
            particles.betas,
        )
