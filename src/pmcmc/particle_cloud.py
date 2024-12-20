from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict

import jax
import numpy as np
from jax import Array, float0
from jax import numpy as jnp
from jax import random
from jax import random as random
from jax.scipy.stats import nbinom as nbinom
from jax.scipy.stats import norm as normal
from jax.typing import ArrayLike

from src.pmcmc.global_settings import GlobalSettings
from src.pmcmc.transition import Transition

KeyArray = Array


@dataclass
class ParticleCloud:
    """Represents a cloud of particles. Includes methods for updating
    particles, computing weights, resampling, and perturbing variables.

    Attributes:
        settings: Global filter settings for initialization.
        states: An (N, S, T) array of system states at each time step, where
            N is number of particles,
            S is size of state vector,
            T is length of time series.
        weights: An (N, T) of particle weights, where
            N is number of particles,
            T is length of time series.
        hosp_estimates: An array of hospital estimates. Length = num_particles.

    Examples:
        weights[i, t] will return the ith particle's weight at time t.
        states[:, 4, t] will return all particles' new_H counts at time t.
    """

    def __init__(
        self,
        settings: GlobalSettings,
        transition: Transition,
        logger: Logger,
        theta: Dict[str, Any],
    ) -> None:
        self.settings = settings
        self.model = transition
        self.logger = logger

        self._apply_mcmc_params(theta=theta)

        seed = 43
        self.key = random.PRNGKey(seed)

        self.states = self.set_initial_states()

        self.weights = jnp.zeros((self.settings.num_particles, self.settings.runtime))
        self.hosp_estimates = jnp.zeros(
            (self.settings.num_particles, self.settings.runtime)
        )
        self.all_resamples = jnp.zeros(
            (
                self.settings.num_particles,
                self.settings.num_particles,
                self.settings.runtime,
            )
        )
        self.likelihoods = jnp.zeros(self.settings.runtime)
        self.betas = jnp.zeros((self.settings.num_particles, self.settings.runtime))

    def set_initial_states(self) -> Array:
        """
        Sets the initial states of the particles,
        according to priors given in `config.toml`.

        Returns:
            An array of initial states.
        """
        self.key, *initial_state_keys = random.split(
            self.key, self.settings.num_particles + 1
        )
        initial_states = jnp.array(
            [self._get_initial_state(k) for k in initial_state_keys]
        )
        states = jnp.zeros(
            (
                self.settings.num_particles,
                initial_states.shape[-1],
                self.settings.runtime,
            )
        )
        states = states.at[:, :, 0].set(initial_states)
        return states

    def _apply_mcmc_params(self, theta: Dict[str, Any]):
        """
        Overrides default params with MCMC proposals.

        Args:
            theta: a dictionary {param_name: value}, passed
                into the PF from MCMC.

        Returns:
            None: Instance parameters are updated directly.
        """
        for key, value in theta.items():
            if hasattr(self.model.params, "update_param"):
                try:
                    self.model.params.update_param(key, value)
                except AttributeError:
                    if key in self.settings.__dict__.keys():
                        setattr(self.settings, key, value)
                    else:
                        raise ValueError(
                            f"Initial_theta has an unrecognized parameter: {key}."
                        )
            else:
                raise ValueError(f"Model does not support parameter updates.")

    def _get_initial_state(self, key: KeyArray) -> Array:
        """
        Gets an initial state for one particle.

        The entire population is susceptible. Then, we draw from uniform
        random to infect some portion of the susceptible population.

        Args:
            key: A JAX PRNG key

        Returns:
            Initial state vector.
        """
        key1, key2 = random.split(key, 2)
        population = self.settings.population

        # state = [S, I, R, H, new_H, beta]
        state = [population, 0, 0, 0, 0, 0]

        # Infect a portion of S compartment
        infected_seed = random.uniform(
            key=key1, minval=1, maxval=self.settings.seed_size * population
        )
        state[1] += infected_seed
        state[0] -= infected_seed

        # Initialize beta based on prior
        beta_prior = self.settings.beta_prior
        initial_beta = random.uniform(
            key=key2, shape=(), dtype=float, minval=beta_prior[0], maxval=beta_prior[1]
        )
        state[5] = initial_beta

        return jnp.array(state)

    def enforce_population_constraint(self, state) -> Array:
        """
        Scales each compartment (S,I,R,H) to ensure that
        the compartments sum to N, the total population.

        This is necessary because the stochastic system causes the sum of
        the compartments to drift away from N.

        Args:
            state: The state vector for some particle.

        Returns:
            State vector where the S,I,R,H compartments
                sum to the total population N.
        """
        S, I, R, H, new_H, beta = state
        total_population = S + I + R + H
        scale = self.settings.population / total_population

        # Scale all compartments to ensure the sum equals N
        S *= scale
        I *= scale
        R *= scale
        H *= scale

        return jnp.array([S, I, R, H, new_H, beta])

    def _update_single_particle(self, state: ArrayLike, t: int) -> Array:
        """
        For a single particle, step the state forward 1 discrete time step.

        Helper function for update_all_particles. Each particle's update is
        performed separately to accommodate for individual gradient/sensitivity
        calculations that may be added in the future.

        Args:
            state: the current state of the particle at time t.
            t: the current time step.

        Returns:
            New state vector for a single particle.
        """
        num_steps = int(1 / self.settings.dt)
        state = state.at[4].set(0)  # set new_hospitalizations to 0.
        for _ in range(num_steps):
            det_update = self.model.det_component(state, t) * self.settings.dt
            state += det_update
            self.key, subkey = random.split(self.key)
            sto_update = self.model.sto_component(state, self.settings.dt, subkey)
            state += sto_update

        # new hospitalizations cannot be less than 0
        state = state.at[4].set(jnp.maximum(state[4], 0))
        state = self.enforce_population_constraint(state)
        return state

    def update_betas(self, t: int) -> None:
        """
        Updates the beta parameter of each particle at time t.

        Args:
            t: current time step

        Returns:
            None. Update is performed in place.
        """
        betas = self.states[:, 5, t - 1]
        self.key, *subkeys = random.split(self.key, num=self.settings.num_particles + 1)
        subkeys = jnp.stack(subkeys)
        updates = jax.vmap(self.model.update_beta, in_axes=(0, None, None, 0))(
            betas, self.settings.dt, t, subkeys
        )
        betas += updates
        self.states = self.states.at[:, 5, t].set(betas)

    def update_all_particles(self, t: int) -> None:
        """Propagate all particle state vectors forward one time step.

        Args:
            t: current time step

        Returns:
            None. This method updates the instance attributes directly.
        """
        self.update_betas(t)

        # Update each particle, given the previous time step's state.
        new_states = jax.vmap(self._update_single_particle, in_axes=(0, None))(
            self.states[:, :, t - 1], t
        )

        # Save the new state at time t.
        self.states = self.states.at[:, :, t].set(new_states)

        # Record hospitalization estimates
        new_hosp_estimates = self.states[:, 4, t]
        self.hosp_estimates = self.hosp_estimates.at[:, t].set(new_hosp_estimates)

    @staticmethod
    @jax.jit
    def _compute_single_weight(
        reported_data: int, particle_estimate: float | ArrayLike, r: float | int
    ) -> Array:
        """
        Computes the un-normalized weight of a single particle.
        Helper function for compute_all_weights.

        Args:
            reported_data: Reported new hospitalization case counts at current time step.
            particle_estimate: Reported new hospitalization case counts at current time step.
            r: dispersion parameter for nbinom likelihood.

        Returns:
            A 1-element JAX array, containing an un-normalized weight for a single particle.
        """
        epsilon = 0.005
        weight = nbinom.logpmf(
            k=reported_data,
            n=r,
            p=r / (r + particle_estimate + epsilon),
        )
        return weight

    def compute_all_weights(self, reported_data: int | float, t: int) -> None:
        """
        Update the weights for every particle. Saves the Monte Carlo
        likelihood estimate for the weights.

        Args:
            reported_data: Reported new hospitalization case counts at time t.
            t: current time step.

        Returns:
            None. Updates the instance weights directly.
        """
        new_weights = jax.vmap(self._compute_single_weight, in_axes=(None, 0, None))(
            reported_data, self.hosp_estimates[:, t], self.settings.dispersion
        )
        self.weights = self.weights.at[:, t].set(new_weights)
        self.save_likelihood(new_weights, t)

    def save_likelihood(self, weights: jnp.ndarray, t: int) -> None:
        """Saves the Monte Carlo estimate of the likelihood at time t.

        Args:
            weights: the particle weights at current time step.
            t: current time step.

        Returns:
            None: This method updates the instance likelihoods directly.
        """
        likelihood = jacobian(weights)[-1] - jnp.log(self.settings.num_particles)
        self.likelihoods = self.likelihoods.at[t].set(likelihood)

    def normalize_weights(self, t: int) -> None:
        """Normalize the weights using the Jacobian algorithm.
        Updates the instance weights directly.

        Args:
            t: current time step

        Returns:
            None. Directly updates the instance weights.
        """
        norm_weights = log_norm(self.weights[:, t])
        self.weights = self.weights.at[:, t].set(norm_weights)

    def resample_deprecated(self, t: int) -> None:
        """
        Deprecated version of the resampling algorithm.
        Moved to a JAX version below.
        """
        resampling_indices = jnp.zeros(self.settings.num_particles, dtype=int)
        cdf_log = jacobian(self.weights[:, t])
        self.key, subkey = random.split(self.key)

        u = random.uniform(
            key=subkey,
            shape=(),
            dtype=float,
            minval=0,
            maxval=1 / self.settings.num_particles,
        ).item()

        i = 0
        for j in range(self.settings.num_particles):
            u_j = jnp.log(u + (j / self.settings.num_particles))

            while i < self.settings.num_particles and u_j > cdf_log[i]:
                i += 1
            resampling_indices = resampling_indices.at[j].set(i)

        self.all_resamples = self.all_resamples.at[:, :, t].set(resampling_indices)
        self.states = self.states.at[:, :, t].set(self.states[resampling_indices, :, t])

    def resample(self, t: int) -> None:
        """
        Systematic resampling algorithm.

        Args:
            t: current time step

        Returns:
            None. Directly updates the instance states and beta values.
        """
        cdf_log = jacobian(self.weights[:, t])
        self.key, subkey = random.split(self.key)
        u = random.uniform(
            key=subkey,
            shape=(),
            dtype=float,
            minval=0,
            maxval=1 / self.settings.num_particles,
        )

        def scan_body(i, j):
            u_j = jnp.log(u + (j / self.settings.num_particles))

            def cond_fun(loop_vars):
                i, _ = loop_vars
                return jnp.logical_and(
                    i < self.settings.num_particles, u_j > cdf_log[i]
                )

            def body_fun(loop_vars):
                i, _ = loop_vars
                return (i + 1, i)

            i, _ = jax.lax.while_loop(cond_fun, body_fun, (i, i))

            return i, i

        _, resampling_indices = jax.lax.scan(
            scan_body, 0, jnp.arange(self.settings.num_particles)
        )

        self.all_resamples = self.all_resamples.at[:, :, t].set(resampling_indices)
        self.states = self.states.at[:, :, t].set(self.states[resampling_indices, :, t])

    def compute_marginal_likelihood(self):
        """Returns the marginal likelihood, to be used by MCMC."""
        sums = jacobian(self.weights[:, -1])
        return sums[-1]

    def perturb_beta(self, t: int):
        """Adds stochastic perturbations to each particle's beta value."""
        betas = self.states[:, 5, t]
        self.key, subkey = random.split(self.key)
        perturbations = random.normal(
            key=subkey, shape=(self.settings.num_particles,), dtype=float
        )
        betas += perturbations * 0.005
        self.states = self.states.at[:, 5, t].set(betas)
        self.betas = self.betas.at[:, t].set(betas)

        # Ensure non-negative beta values
        self.betas = jnp.where(self.betas > 0, self.betas, 0.001)


def jacobian(input_array: ArrayLike) -> Array:
    """
    The Jacobian algorithm, used in log likelihood normalization and
    resampling processes.

    Args:
        input_array: An array of values to sum.

    Returns:
        delta: The vector of partial sums of the input array.
    """
    n = len(input_array)
    delta = jnp.zeros(n)
    delta = delta.at[0].set(input_array[0])
    for i in range(1, n):
        delta_i = jnp.max(jnp.array([input_array[i], delta[i - 1]])) + jnp.log(
            1 + jnp.exp(-1 * jnp.abs(delta[i - 1] - input_array[i]))
        )
        delta = delta.at[i].set(delta_i)
    return delta


def log_norm(log_weights: ArrayLike) -> Array:
    """
    Normalizes the probability space using the Jacobian algorithm as
    defined in jacobian().

    The Jacobian outputs an array of partial sums, where the
    last element is the sum of all inputs. Thus, the normalization
    factor is this last element.

    Args:
        log_weights: An array of length num_particles.
            Contains the log-weight for each particle.

    Returns:
        Array. Array[p] is the normalized log weight for particle p.
    """
    normalization_factor = jacobian(log_weights)[-1]
    log_weights -= normalization_factor
    return log_weights
