from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as random
from jax import Array
from jax.numpy.linalg import cholesky
from jax.typing import ArrayLike
from tqdm.notebook import tqdm

from src.particle_filter.filter_algo import PFOutput
from src.particle_filter.initialize_filter import initialize_particle_filter
from src.particle_filter.logger import get_logger
from src.particle_filter.prior import Prior
from src.particle_filter.logger import get_logger


class PMCMC:
    def __init__(
        self,
        iterations: int,
        init_theta: Dict,
        prior: Prior,
        location_info: Dict[str, Any],
        observation_data: ArrayLike,
    ) -> None:
        """Initializes a Particle MCMC algorithm object.

        Args:
            iterations: Number of MCMC iterations.
            init_theta: Initial proposal for theta, the parameter vector that will be passed into the particle filter.
            prior: Prior distribution function.
            location_info: A dictionary containing information about the current location. See `main.py` for contents.
            observation_data: An array of reported data from the CDC.
        """
        self._num_params = len(init_theta)
        self._iterations = iterations
        self._prior = prior
        self._key = random.PRNGKey(47)
        self.logger = get_logger()
        self.location_settings = location_info
        self.observation_data = observation_data

        self._mle_betas = None
        self._mle_hospitalizations = None
        self._mle_states = None
        self._max_likelihood = float("-inf")

        self._thetas = jnp.zeros((self._num_params, iterations))
        self._likelihoods = jnp.zeros(iterations)
        self._accept_record = jnp.zeros(iterations)

        self._mu = jnp.zeros(self._num_params)
        self._cov = jnp.eye(self._num_params)

        self._thetas = self._thetas.at[:, 0].set(jnp.array(list(init_theta.values())))
        # TODO: Fix the disgusting array hack on these surrounding lines
        self._likelihoods = self._likelihoods.at[0].set(prior.get_likelihood(self._thetas[:, 0]))

        self.theta_dictionary_template = init_theta

        if jnp.isfinite(self._likelihoods[0]):
            # Set the initial MLE estimates using initial theta.
            pf_output = self._run_filter(theta_proposal=init_theta)
            self._likelihoods = self._likelihoods.at[0].set(jnp.sum(pf_output.likelihood))
            self.update_new_mle(
                        new_likelihood=self._likelihoods[0],
                        particle_estimates=pf_output.hosp_estimates,
                        particle_states=pf_output.states,
                        particle_betas=pf_output.betas,
                    )

    def run(self) -> None:
        """Runs the MCMC algorithm.

        At each iteration, we propose a new parameter vector `theta`.
        The algo runs a particle filter with this new theta (`self.run_filter`).
        The PF returns a likelihood (amongst other data).
        This new likelihood is compared against the previous theta using the Metropolis-Hastings algorithm.
        If accepted, we move to the new theta. If rejected, we stay at our current location.

        Returns:
            None. Quantities of interest are accessible via the instance attributes.
        """
        for i in tqdm(range(1, self._iterations), desc="PMCMC Progress", colour="MAGENTA"):
            theta_prev = self._thetas[:, i - 1]
            theta_prop = self.generate_theta_proposal(previous_theta=theta_prev)

            proposal_likelihood = self._prior.get_likelihood(theta_prop)

            if jnp.isfinite(proposal_likelihood):
                pf_output = self._run_filter(theta_proposal=theta_prop)
                proposal_likelihood += jnp.sum(pf_output.likelihood)

                if proposal_likelihood > self._max_likelihood:
                    self.update_new_mle(
                        new_likelihood=proposal_likelihood,
                        particle_estimates=pf_output.hosp_estimates,
                        particle_states=pf_output.states,
                        particle_betas=pf_output.betas,
                    )

                self.accept_reject(theta_prop, proposal_likelihood, i)

            else:
                # Reject automatically, because the ratio is negative infinity.
                self.reject_proposal(i)

            # TODO: Implement covariance update
            # self.update_cov(i)

    def generate_theta_proposal(self, previous_theta):
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
        self._key, subkey = random.split(self._key)
        random_params = random.normal(key=subkey, shape=(self._num_params))
        cholesky_matrix = cholesky((2.38**2 / self._num_params) * self._cov)
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
            logger=self.logger
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
            new_theta_dict[key] = theta[index].item()
        return new_theta_dict

    def accept_reject(
        self, theta_prop: ArrayLike, new_likelihood: float, iteration: int
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
            new_likelihood - self._likelihoods[iteration - 1]
        ).item()
        acceptance_probability = min(1, acceptance_probability)
        self._key, subkey = random.split(self._key)
        u = random.uniform(key=subkey, minval=0, maxval=1).item()
        if jnp.log(u) < acceptance_probability:
            self.accept_proposal(theta_prop, new_likelihood, iteration)
        else:
            self.reject_proposal(iteration)

    def accept_proposal(
        self, theta_prop: ArrayLike, new_likelihood: float, iteration: int
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
        self._thetas = self._thetas.at[:, iteration].set(theta_prop)
        self._likelihoods = self._likelihoods.at[iteration].set(new_likelihood)
        self._accept_record = self._accept_record.at[iteration].set(1)

    def reject_proposal(self, i: int):
        """
        Reject the proposed theta.

        Instance attributes are updated in place.
        The previous theta and likelihood are copied to the current iteration.
        """
        self._thetas = self._thetas.at[:, i].set(self._thetas[:, i - 1])
        self._likelihoods = self._likelihoods.at[i].set(self._likelihoods[i - 1])

    def update_cov(self, current_iter) -> None:
        pass

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

    @property
    def mle_betas(self):
        """
        Get the beta values from the maximum likelihood run of the particle filter.
        """
        return self._mle_betas

    @property
    def mle_hospitalizations(self):
        """
        Get the hospitalization estimates from the maximum likelihood
        run of the particle filter.
        """
        return self._mle_hospitalizations

    @property
    def mle_states(self):
        """
        Get the state vectors at each time step from the maximum likelihood run of the particle filter.
        """
        return self._mle_states

    def log_config_file(self, config_file_path):
        """Logs the contents of the config.toml file."""
        logger = get_logger()

        # Read the configuration file
        with open(config_file_path, "r") as file:
            config_contents = file.read()

        # Log the contents of the configuration file
        logger.info("Logging configuration file contents:")
        logger.info(config_contents)
