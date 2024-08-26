import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.random import PRNGKey

from src.pmcmc.transition import OUModel

days = 80
step_beta_switch = 40

initial_beta = 0.25
final_beta = 0.15

# Create a linear decline from initial_beta to final_beta
linear_decline = np.linspace(initial_beta, final_beta, step_beta_switch)

# Combine the declining part with the final constant value
step_beta = np.concatenate([linear_decline, [final_beta] * (days - step_beta_switch)])

pop = 1000000
infected = 0.01 * pop
susceptible = pop - infected
initial_state = jnp.array([susceptible, infected, 0, 0, 0, 0.3])
key = PRNGKey(0)

import os

from src import paths

config_path = os.path.join(paths.PMCMC_DIR, "config.toml")

ou_model = OUModel(config_path)

det_output = [initial_state.copy()]
case_reports = [0]


def det_update(state, time_step):
    state = state.at[5].set(step_beta[time_step])
    update = ou_model.det_component(state, time_step)
    case_reports.append(update[4].item())
    state += update
    return state


for t in range(1, days):
    det_output.append(det_update(det_output[-1], t))

import numpy as np
from scipy.stats import nbinom

case_reports = np.asarray(case_reports)
# Define the dispersion parameter r
r = 10.0  # Adjust as needed

# Calculate the probability p for each time point
p = r / (r + jnp.maximum(case_reports, 1e-5))

num_draws = case_reports.shape

noisy_case_reports = nbinom.rvs(n=r, p=p, size=num_draws)

from src.pmcmc.pmcmc_multi import PMCMC
from os import path
import toml
from src.pmcmc.prior import UniformPrior

location_info = {
    "population": pop,
    "location_code": "04",
    "target_date": "2047-10-28",
    "runtime": days,
}

config_path = path.join(paths.PMCMC_DIR, "config.toml")
config = toml.load(config_path)

prior = UniformPrior()

pmcmc_algo = PMCMC(
    iterations=config["mcmc"]["iterations"],
    init_thetas=config["mcmc"]["initial_theta"],
    prior=prior,
    location_info=location_info,
    observation_data=noisy_case_reports,
    burn_in=config["mcmc"]["burn_in"],
)

pmcmc_algo.run()
