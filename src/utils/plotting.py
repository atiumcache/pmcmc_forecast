import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
from src.utils import paths
import os


def plot_mcmc_overview(file_path: str) -> None:
    """
    Provides an overview of MCMC performance.

    Args:
        file_path: An absolute path to a json file.

    Outputs two plots:
    - theta over iterations
    - likelihood over iterations

    Prints basic diagnostic info.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    theta_chains = np.array(data["theta_chains"])
    likelihoods = np.array(data["likelihoods"])
    accept_records = np.array(data["accept_record"])
    iterations = data["iteration"]  # Assuming iterations are consistent for all chains

    accept_rates = [
        round(chain.sum() / len(chain), ndigits=3)
        for chain in accept_records[:, :iterations]
    ]
    for chain, rate in enumerate(accept_rates):
        print(f"Chain {chain + 1}: {rate} acceptance rate.")

    # Plot Theta Chains
    plt.figure(figsize=(10, 6))
    for i in range(theta_chains.shape[0]):
        sns.lineplot(
            x=range(iterations),
            y=theta_chains[i, 0, :iterations],
            label=f"Theta Chain {i+1}",
        )
    plt.title("Theta Chains Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Theta Value")
    plt.legend()
    plt.show()

    # Plot Likelihoods
    plt.figure(figsize=(10, 6))
    for i in range(likelihoods.shape[0]):
        sns.lineplot(
            x=range(iterations),
            y=likelihoods[i, :iterations],
            label=f"Likelihood Chain {i+1}",
        )
    plt.title("Likelihoods Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.legend()
    plt.show()


def plot_predictions_with_quantile_range(prediction_date, location, weeks_prior=4):
    """
    Plot the actual hospitalization reports against the predicted hospitalizations.

    Args:
        prediction_date: the date that we forecast from.
        location: 2 digit location code.
        weeks_prior: number of weeks to plot data prior to prediction_data.

    Returns:
        None. Outputs a plot.
    """
    pred_path = os.path.join(
        paths.HOSP_OUTPUT_DIR, prediction_date, f"{location}-PMCMC-flu-predictions.csv"
    )
    hosp_csv_path = os.path.join(
        paths.DATASETS_DIR, "hosp_data", f"hosp_{location}.csv"
    )
    # Load your prediction and hospitalization data
    prediction_data = pd.read_csv(pred_path)
    actual_data = pd.read_csv(hosp_csv_path)

    # Convert the necessary date columns to datetime
    actual_data["date"] = pd.to_datetime(actual_data["date"])
    prediction_data["target_end_date"] = pd.to_datetime(
        prediction_data["target_end_date"]
    )
    prediction_data["reference_date"] = pd.to_datetime(
        prediction_data["reference_date"]
    )

    # Convert prediction_date to datetime
    prediction_date = pd.to_datetime(prediction_date)

    # Filter actual data for the desired location
    actual_data_filtered = actual_data

    # Filter actual hospitalizations for the 4 weeks prior to the prediction_date
    start_date = prediction_date - pd.DateOffset(weeks=weeks_prior)
    actual_subset = actual_data_filtered[
        (actual_data_filtered["date"] >= start_date)
        & (actual_data_filtered["date"] <= prediction_date + pd.DateOffset(weeks=4))
    ]

    # Get predictions for the same location and prediction date
    prediction_subset = prediction_data[
        (prediction_data["reference_date"] == prediction_date)
    ]

    # Select relevant quantiles directly
    lower_quantile_data = prediction_subset[prediction_subset["output_type_id"] == 0.05]
    median_quantile_data = prediction_subset[
        prediction_subset["output_type_id"] == 0.50
    ]
    upper_quantile_data = prediction_subset[
        prediction_subset["output_type_id"] == 0.95
    ]  # Resample actual data to weekly sums
    actual_weekly = actual_subset.resample("W-Sat", on="date").sum().reset_index()

    # Plot setup
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot actual hospitalizations (weekly sums)
    ax.plot(
        actual_weekly["date"],
        actual_weekly["previous_day_admission_influenza_confirmed"],
        label="Actual Hospitalizations (Weekly Sum)",
        color="black",
        marker="o",
    )

    actual_on_pred_date = actual_weekly.loc[
        actual_weekly["date"] == prediction_date,
        "previous_day_admission_influenza_confirmed",
    ].values[0]

    lower_quantile_data = lower_quantile_data.drop(
        columns=[
            "horizon",
            "location",
            "output_type_id",
            "output_type",
            "reference_date",
        ]
    )
    new_row = {
        "target_end_date": pd.to_datetime(prediction_date),
        "value": actual_on_pred_date,
    }
    lower_quantile_data.reset_index(drop=True, inplace=True)
    upper_quantile_data.reset_index(drop=True, inplace=True)
    lower_quantile_data.loc[4] = new_row
    upper_quantile_data.loc[4] = new_row
    lower_quantile_data = lower_quantile_data.sort_values(by="target_end_date")
    upper_quantile_data = upper_quantile_data.sort_values(by="target_end_date")

    # Plot the quantile range (5th and 95th percentiles) as a shaded area
    ax.fill_between(
        lower_quantile_data["target_end_date"],
        lower_quantile_data["value"],
        upper_quantile_data["value"],
        color="blue",
        alpha=0.2,
        label="5th-95th Percentile Range",
    )

    # Plot the median (50th percentile) as a line
    ax.plot(
        median_quantile_data["target_end_date"],
        median_quantile_data["value"],
        label="50th Percentile (Median)",
        color="blue",
    )
    ax.plot(
        median_quantile_data["target_end_date"],
        median_quantile_data["value"],
        "s",
        color="blue",
        markersize=6,
    )
    one_week_ahead = prediction_date + pd.DateOffset(weeks=1)
    ax.plot(
        [prediction_date, one_week_ahead],
        [
            actual_on_pred_date,
            median_quantile_data.loc[
                median_quantile_data["target_end_date"] == one_week_ahead, "value"
            ].values[0],
        ],
    )

    # Add a vertical dashed line for the prediction date
    ax.axvline(x=prediction_date, color="red", linestyle="--", label="Forecast Date")
    # Formatting the date on the x-axis
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Hospitalizations (Weekly Sum)")
    plt.title(
        f'Actual vs Predicted Hospitalizations | Location {location} | Forecast Date: {prediction_date.strftime("%Y-%m-%d")}'
    )
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
