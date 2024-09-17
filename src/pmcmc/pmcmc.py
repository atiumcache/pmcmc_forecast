import json
import os
from datetime import datetime
from os import path
from typing import Any, Dict, List

import jax.numpy as jnp
import jax.random as random
import \
    pytz
from jax import Array, vmap
from jax.numpy.linalg import cholesky
from jax.typing import ArrayLike
from tqdm import tqdm
import pandas as pd

from src import paths
from src.pmcmc.filter_algo import PFOutput
from src.pmcmc.initialize_filter import initialize_particle_filter
from src.pmcmc.logger import get_logger
from src.pmcmc.prior import Prior


class PMCMC:
    def __init__(
        self,
        iterations: int,
        burn_in: int,
        init_thetas: List[Dict],
        prior: Prior,
        location_info: Dict[str, Any],
        observation_data: ArrayLike,
        num_chains: int = 3,
    ) -> None:
        self._num_params = len(init_thetas[0])
        self._iterations = iterations
        self._prior = prior
        self._key = random.PRNGKey(47)
        self.logger = get_logger()
        self.location_settings = location_info
        self.observation_data = observation_data
        self.num_chains = num_chains
        self.burn_in = burn_in

        self._mle_betas = None
        self._mle_hospitalizations = None
        self._mle_states = None
        self._max_likelihood = float("-inf")

        self._theta_chains = jnp.zeros((self.num_chains, self._num_params, iterations))
        self._likelihoods = jnp.zeros((self.num_chains, iterations))
        self._accept_record = jnp.zeros((self.num_chains, iterations))

        self._mu = jnp.zeros(self._num_params)
        self._cov = jnp.eye(self._num_params)

        for chain_idx, init_theta in enumerate(init_thetas):
            init_theta_values = jnp.array(list(init_theta.values()))
            self._theta_chains = self._theta_chains.at[chain_idx, :, 0].set(
                init_theta_values
            )
            self._likelihoods = self._likelihoods.at[chain_idx, 0].set(
                prior.get_likelihood(init_theta_values)
            )

            if jnp.isfinite(self._likelihoods[chain_idx, 0]):
                pf_output = self._run_filter(theta_proposal=init_theta)
                self._likelihoods = self._likelihoods.at[chain_idx, 0].set(
                    jnp.sum(pf_output.likelihood)
                )
                self.update_new_mle(
                    new_likelihood=self._likelihoods[chain_idx, 0],
                    particle_estimates=pf_output.hosp_estimates,
                    particle_states=pf_output.states,
                    particle_betas=pf_output.betas,
                )

        self.theta_dictionary_template = init_thetas[0]

        json_dir = path.join(paths.PMCMC_RUNS_DIR, location_info["location_code"])
        os.makedirs(json_dir, exist_ok=True)
        self.json_out_path = path.join(json_dir, f"{location_info['target_date']}.json")
        self.init_json_output_file()

    def run(self) -> None:
        """
        Runs the MCMC algorithm.

        At each iteration, we propose a new parameter vector `theta`.
        The algo runs a particle filter with this new theta (`self.run_filter`).
        The PF returns a likelihood (amongst other data).
        This new likelihood is compared against the previous theta using the Metropolis-Hastings algorithm.
        If accepted, we move to the new theta. If rejected, we stay at our current location.

        Returns:
            None. Quantities of interest are accessible via the instance attributes.
        """
        for i in tqdm(
            range(1, self._iterations), desc="PMCMC Progress", colour="MAGENTA"
        ):

            self._key, subkey = random.split(self._key)
            subkeys = random.split(subkey, self.num_chains)
            theta_prev = self._theta_chains[:, :, i - 1]
            theta_prop = vmap(self.generate_theta_proposal)(theta_prev, subkeys)

            proposal_likelihood = vmap(self._prior.get_likelihood)(theta_prop)

            valid_proposals = jnp.isfinite(proposal_likelihood)
            pf_outputs = vmap(self._run_filter)(theta_prop[valid_proposals])

            proposal_likelihood = proposal_likelihood.at[valid_proposals].set(
                proposal_likelihood[valid_proposals]
                + jnp.sum(pf_outputs.likelihood, axis=1)
            )

            for chain in range(self.num_chains):
                if proposal_likelihood[chain] > self._max_likelihood:
                    self.update_new_mle(
                        new_likelihood=proposal_likelihood[chain],
                        particle_estimates=pf_outputs.hosp_estimates[chain],
                        particle_states=pf_outputs.states[chain],
                        particle_betas=pf_outputs.betas[chain],
                    )
                    self.output_data(in_progress=True)
                self.accept_reject(
                    theta_prop[chain], proposal_likelihood[chain], i, chain
                )
                if i % 10 == 0:
                    self.log_status(iteration=i, theta=theta_prop[chain], chain=chain)

            # TODO: Diagnose and fix R-Hat convergence check
            """
            if i % 10 == 0:
                if self.chains_converged():
                    self.logger.info(
                        f"Chains converged at iteration {i}, according to R_hat convergence metric."
                    )
                    break
            """
            self.save_state_to_json(i)

            # TODO: Implement covariance update
            # self.update_cov(i)

    def output_data(self, in_progress: bool = False) -> None:
        """
        Saves data to CSV for analysis and/or later use.

        Args:
            in_progress: Indicates that we are saving data while
            the algorithm is still running. So we save to a different
            file vs. the final output.
        """
        f_string = ""
        if in_progress:
            f_string = "in_progress_"

        # Get current time in MST
        mst = pytz.timezone('MST')
        current_time = datetime.now(mst)
        current_date = current_time.strftime('%Y-%m-%d_%H-%M')

        loc_code: str = self.location_settings['location_code']
        files_dir: str = path.join(paths.PMCMC_RUNS_DIR, loc_code)
        mle_betas_path: str = path.join(files_dir, f'{f_string}mle_betas-{current_date}.csv')
        mle_states_path: str = path.join(files_dir, f'{f_string}mle_states-{current_date}.npy')
        likelihoods_path: str = path.join(files_dir, f'{f_string}likelihoods{current_date}.npy')
        thetas_path: str = path.join(files_dir, f'{f_string}thetas-{current_date}.npy')
        acceptance_path: str = path.join(files_dir, f'{f_string}acceptance-{current_date}.csv')

        betas_df = pd.DataFrame(self._mle_betas)
        betas_df.to_csv(mle_betas_path)

        jnp.save(file=mle_states_path, arr=self._mle_states)
        jnp.save(file=likelihoods_path, arr=self._likelihoods)
        jnp.save(file=thetas_path, arr=self._theta_chains)
        jnp.save(file=acceptance_path, arr=self._accept_record)

    def generate_theta_proposal(self, previous_theta, key):
        """
        Generate a proposal for the next iteration of the Markov Chain.

        This method generates a new proposed value for the parameter vector `theta` by
        adding a random perturbation to the previous parameter vector.
        The perturbation is drawn from a multivariate normal distribution
        with a covariance matrix scaled by the Cholesky decomposition. The scaling
        factor is derived from the rule-of-thumb 2.38^2 / d, where d is the number of parameters.

        Args:
            previous_theta: the parameter vector from the previous iteration

        Returns:
            theta_proposal: the proposed parameter vector for the current iteration.
        """
        random_params = random.normal(key=key, shape=(self._num_params,))
        cholesky_matrix = cholesky((1.2**2 / self._num_params) * self._cov)
        theta_proposal = previous_theta + cholesky_matrix @ random_params
        return theta_proposal

    def _run_filter(self, theta_proposal: ArrayLike | Dict) -> PFOutput:
        """
        Run the particle filter using the proposed parameters.

        Args:
            theta_proposal: the proposed parameter vector.

        Returns:
            PFOutput object. Output data is accessible via instance attributes.
        """
        if not isinstance(theta_proposal, Dict):
            theta_proposal = self._convert_theta_to_dict(theta_proposal)

        pf_algo = initialize_particle_filter(
            state_population=self.location_settings["population"],
            location_code=self.location_settings["location_code"],
            target_date=self.location_settings["target_date"],
            runtime=self.location_settings["runtime"],
            logger=self.logger,
        )
        pf_output = pf_algo.run(
            observation_data=self.observation_data, theta=theta_proposal
        )
        return pf_output

    def _convert_theta_to_dict(self, theta: ArrayLike) -> Dict[str, float]:
        """
        Convert a theta vector into a dictionary mapping parameter names to values.

        This is useful for passing parameters to the particle filter
        without manually matching indices.

        Returns:
            Dictionary containing the parameter values.
        """
        new_theta_dict = {}
        for index, key in enumerate(self.theta_dictionary_template):
            new_theta_dict[key] = theta[index]
        return new_theta_dict

    def accept_reject(
        self, theta_prop: ArrayLike, new_likelihood: float, iteration: int, chain: int
    ) -> None:
        """
        Metropolis-Hastings algorithm to determine if the proposed theta
        is accepted or rejected.

        Args:
            theta_prop: the proposed theta vector
            new_likelihood: the likelihood of theta_prop
            iteration: the current MCMC iteration

        Returns:
            None: This method modifies instance attributes in place.
        """
        acceptance_probability = (
            new_likelihood - self._likelihoods[chain, iteration - 1]
        )
        acceptance_probability = jnp.minimum(1, acceptance_probability)
        self._key, subkey = random.split(self._key)
        u = random.uniform(key=subkey, minval=0, maxval=1)
        if jnp.log(u) < acceptance_probability:
            self.accept_proposal(theta_prop, new_likelihood, iteration, chain)
        else:
            self.reject_proposal(iteration, chain)

    def accept_proposal(
        self, theta_prop: ArrayLike, new_likelihood: float, iteration: int, chain: int
    ) -> None:
        """
        Accept the proposed theta.

        Args:
            theta_prop: the proposed theta vector
            new_likelihood: the likelihood of theta_prop
            iteration: the current MCMC iteration

        Returns:
            None: This method modifies instance attributes in place.
        """
        self._theta_chains = self._theta_chains.at[chain, :, iteration].set(theta_prop)
        self._likelihoods = self._likelihoods.at[chain, iteration].set(new_likelihood)
        self._accept_record = self._accept_record.at[chain, iteration].set(1)

    def reject_proposal(self, i: int, chain: int) -> None:
        """
        Reject the proposed theta.

        Instance attributes are updated in place.
        The previous theta and likelihood are copied to the current iteration.
        """
        self._theta_chains = self._theta_chains.at[chain, :, i].set(
            self._theta_chains[chain, :, i - 1]
        )
        self._likelihoods = self._likelihoods.at[chain, i].set(
            self._likelihoods[chain, i - 1]
        )

    def update_cov(self, current_iter) -> None:
        """
        Updates the covariance matrix, which is used to generate theta proposals.
        """
        raise NotImplementedError("This method has not yet been implemented.")

    def update_new_mle(
        self, new_likelihood, particle_estimates, particle_states, particle_betas
    ):
        """
        Updates the MLE based on the new likelihood and PF output.
        """
        self._max_likelihood = new_likelihood
        self._mle_hospitalizations = particle_estimates
        self._mle_states = particle_states
        self._mle_betas = particle_betas

    def gelman_rubin_diagnostic(self) -> Array:
        """
        Calculate the Gelman-Rubin diagnostic ("R Hat") for each parameter.

        Returns:
            Array: Gelman-Rubin diagnostic for each parameter
        """
        num_chains, num_params, num_iterations = self._theta_chains.shape

        chain_means = jnp.mean(self._theta_chains, axis=2)
        chain_variances = jnp.var(self._theta_chains, axis=2, ddof=1)
        mean_of_chain_means = jnp.mean(chain_means, axis=0)

        # Calculate the between-chain variance
        B = num_iterations * jnp.var(chain_means, axis=0, ddof=1)

        # Calculate the within-chain variance
        W = jnp.mean(chain_variances, axis=0)

        # Calculate the potential scale reduction factor (PSRF)
        var_hat = ((num_iterations - 1) / num_iterations) * W + (1 / num_iterations) * B
        R_hat = jnp.sqrt(var_hat / W)

        return R_hat

    def chains_converged(self) -> bool:
        """
        Check for chain convergence using Gelman-Rubin diagnostic.

        Returns:
            bool: True if chains have converged. False otherwise.
        """
        r_hat = self.gelman_rubin_diagnostic()
        self.logger.info(f"R_hat values: {r_hat}")

        convergence_threshold = 1.1
        return jnp.all(r_hat < convergence_threshold).item()

    @property
    def mle_betas(self):
        """
        Get the beta values from the maximum likelihood run of the particle filter.
        """
        return self._mle_betas

    @property
    def mle_hospitalizations(self):
        """
        Get the hospitalization estimates from the maximum likelihood run of the particle filter.
        """
        return self._mle_hospitalizations

    @property
    def mle_states(self):
        """
        Get the state vectors from the maximum likelihood run of the particle filter.
        """
        return self._mle_states

    def log_config_file(self, config_file_path) -> None:
        """
        Log the contents of the config.toml file.
        """
        logger = self.logger
        with open(config_file_path, "r") as file:
            config_contents = file.read()
        logger.info("Logging configuration file contents:")
        logger.info(config_contents)

    def log_status(self, iteration: int, theta: Array | Dict, chain: int) -> None:
        """
        Log the current status of the algorithm.
        This includes acceptance ratio and current thetas.
        """
        accept_ratio = round(
            sum(self._accept_record[chain, :iteration]) / iteration, ndigits=2
        )
        theta = self._convert_theta_to_dict(theta)
        theta_string_list = [f"{key}: {value}" for key, value in theta.items()]
        theta_string = "".join(theta_string_list)
        self.logger.info(
            f"Iter: {iteration} | Chain: {chain} | Accept Ratio: {accept_ratio} | {theta_string}"
        )

    def init_json_output_file(self) -> None:
        """Overwrite output file if it exists."""
        if path.exists(self.json_out_path):
            with open(self.json_out_path, "w") as f:
                json.dump([], f)

    def save_state_to_json(self, iteration: int) -> None:
        state = {
            "iteration": iteration,
            "theta_chains": self._theta_chains.tolist(),
            "likelihoods": self._likelihoods.tolist(),
            "accept_record": self._accept_record.tolist()[:iteration],
        }
        with open(self.json_out_path, "w") as f:
            json.dump(state, f)
