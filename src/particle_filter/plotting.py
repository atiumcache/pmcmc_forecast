import matplotlib.pyplot as plt
import numpy as np


def prepare_real_data(generated_data, true_beta):
    """Prepares the true data array."""
    real_data = np.array(generated_data).T
    real_data[5, :] = true_beta
    return real_data


def compute_quantiles(data, quantiles):
    """Computes the specified quantiles for the given data."""
    return np.percentile(data, [q * 100 for q in quantiles], axis=0)


def plot_hosp_estimates_vs_true(T, quantile_hosp_estimates, case_reports):
    """Plots the hospital estimates against the true case reports."""
    plt.figure(figsize=(10, 6))
    plt.title("Case Reports: Particle Estimates vs. True")
    plt.fill_between(
        np.arange(T),
        quantile_hosp_estimates[0],
        quantile_hosp_estimates[-1],
        color="blue",
        alpha=0.1,
    )
    plt.fill_between(
        np.arange(T),
        quantile_hosp_estimates[1],
        quantile_hosp_estimates[-2],
        color="blue",
        alpha=0.3,
    )
    plt.plot(np.arange(T), quantile_hosp_estimates[2], color="red")  # Median
    plt.plot(np.arange(T), case_reports, color="black", linestyle="--")  # True data
    plt.xlabel("Time Step")
    plt.ylabel("Hospital Estimates")
    plt.show()


def plot_state_variables(states, true_data, quantiles, state_labels):
    """Plots each state variable separately with quantiles."""
    N, S, T = states.shape

    for i in range(S):
        if i == 4:  # Ignore new_H compartment
            continue

        variable_data = states[:, i, :]  # Shape (N, T)
        quantile_values = compute_quantiles(variable_data, quantiles)

        plt.figure(figsize=(10, 6))
        plt.fill_between(
            np.arange(T),
            quantile_values[0],
            quantile_values[-1],
            color="blue",
            alpha=0.1,
        )
        plt.fill_between(
            np.arange(T),
            quantile_values[1],
            quantile_values[-2],
            color="blue",
            alpha=0.3,
        )
        plt.plot(np.arange(T), quantile_values[2], color="red")  # Median
        plt.plot(
            np.arange(T), true_data[i, :], color="black", linestyle="--"
        )  # True data

        plt.title(state_labels[i])
        plt.xlabel("Time Step")
        plt.ylabel(state_labels[i])
        plt.tight_layout()
        plt.show()
        plt.close()


def generate_plots(
    generated_data, true_beta, particle_estimates, reported_hosp, pf_states
) -> None:
    """Main function to orchestrate the data preparation and plotting."""
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    state_labels = ["S", "I", "R", "H", "new_H", "beta"]

    true_data = prepare_real_data(generated_data, true_beta)
    quantile_hosp_estimates = compute_quantiles(particle_estimates, quantiles)

    # Plot hospital estimates vs. true case reports
    plot_hosp_estimates_vs_true(
        pf_states.shape[2], quantile_hosp_estimates, reported_hosp
    )

    # Plot state variables
    plot_state_variables(pf_states, true_data, quantiles, state_labels)
