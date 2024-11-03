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


import streamlit as st


def plot_daily_predictions(
    prediction_date: str,
    location: str,
    hosp_est_file_name: str,
    weeks_prior: int,
    pf_uncertainty: bool,
    streamlit: bool,
):
    """
    plot...
    """
    raise NotImplementedError


def plot_predictions_with_quantile_range(
    prediction_date: str,
    location: str,
    hosp_est_file_name: str,
    weeks_prior: int,
    daily_resolution: bool = True,
    pf_uncertainty: bool = True,
    streamlit: bool = False,
):
    """
    Plot the actual hospitalization reports against the predicted hospitalizations.

    Args:
        prediction_date: the date that we forecast from.
        location: 2 digit location code.
        weeks_prior: number of weeks to plot data prior to prediction_data.
        pf_uncertainty: if true, plots the particle filter's uncertainty band for hospitalization estimates.
        streamlit: if true, use Streamlit for rendering the plot.

    Returns:
        None. Outputs a plot (either in normal Jupyter/Matplotlib or in Streamlit).
    """
    if daily_resolution:
        plot_daily_predictions(
            prediction_date=prediction_date,
            location=location,
            hosp_est_file_name=hosp_est_file_name,
            weeks_prior=weeks_prior,
            pf_uncertainty=pf_uncertainty,
            streamlit=streamlit,
        )
        return

    # File paths
    pred_path = os.path.join(
        paths.HOSP_OUTPUT_DIR, prediction_date, f"{location}-PMCMC-flu-predictions.csv"
    )
    hosp_csv_path = os.path.join(
        paths.DATASETS_DIR, "hosp_data", f"hosp_{location}.csv"
    )

    # Load data
    prediction_data = pd.read_csv(pred_path)
    actual_data = pd.read_csv(hosp_csv_path)

    # Convert dates to datetime
    actual_data["date"] = pd.to_datetime(actual_data["date"])
    prediction_data["target_end_date"] = pd.to_datetime(
        prediction_data["target_end_date"]
    )
    prediction_data["reference_date"] = pd.to_datetime(
        prediction_data["reference_date"]
    )
    prediction_date = pd.to_datetime(prediction_date)

    # Filter and subset data
    start_date = prediction_date - pd.DateOffset(weeks=weeks_prior)
    actual_subset = actual_data[
        (actual_data["date"] >= start_date)
        & (actual_data["date"] <= prediction_date + pd.DateOffset(weeks=4))
    ]

    prediction_subset = prediction_data[
        prediction_data["reference_date"] == prediction_date
    ]

    # Select relevant quantiles
    lower_quantile_data = prediction_subset[prediction_subset["output_type_id"] == 0.05]
    median_quantile_data = prediction_subset[
        prediction_subset["output_type_id"] == 0.50
    ]
    upper_quantile_data = prediction_subset[prediction_subset["output_type_id"] == 0.95]

    # Resample actual data to weekly sums
    actual_weekly = actual_subset.resample("W-Sat", on="date").sum().reset_index()

    # Create plot
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

    # Interpolation for quantiles
    actual_on_pred_date = actual_weekly.loc[
        actual_weekly["date"] == prediction_date,
        "previous_day_admission_influenza_confirmed",
    ].values[0]

    # Adjusting lower and upper quantile data
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

    # Plot uncertainty range (5th-95th percentiles)
    ax.fill_between(
        lower_quantile_data["target_end_date"],
        lower_quantile_data["value"],
        upper_quantile_data["value"],
        color="orange",
        alpha=0.5,
        label="90% Forecast CI",
    )

    # Plot the median (50th percentile)
    ax.plot(
        median_quantile_data["target_end_date"],
        median_quantile_data["value"],
        label="Median Forecast",
        color="orange",
    )
    ax.plot(
        median_quantile_data["target_end_date"],
        median_quantile_data["value"],
        "s",
        color="purple",
        markersize=5,
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

    # Add vertical line for prediction date
    ax.axvline(x=prediction_date, color="red", linestyle="--", label="Forecast Date")

    # ------------------------------------------------------
    # Plot new hospitalizations uncertainty for PF estimates
    # ------------------------------------------------------
    if pf_uncertainty:
        base_dir = os.path.join(paths.OUTPUT_DIR, "pmcmc_runs")
        loc_dir = os.path.join(base_dir, location)
        mle_states_path = os.path.join(loc_dir, hosp_est_file_name)
        states_np = np.load(mle_states_path)
        last_date_index = abs((prediction_date - pd.to_datetime("2023-06-25")).days)
        start_index = last_date_index - (weeks_prior * 7)
        new_h = states_np[:, start_index:last_date_index]
        new_h_df = pd.DataFrame(new_h)
        full_date_range = pd.date_range(
            start=start_date, periods=new_h_df.shape[1], freq="D"
        )
        new_h_df_T = new_h_df.T
        quantiles = new_h_df_T.quantile([0.025, 0.16, 0.5, 0.84, 0.975], axis=1)
        quantiles = quantiles.T
        quantiles["date"] = full_date_range
        quantiles_subset = quantiles[
            (quantiles["date"] >= start_date) & (quantiles["date"] <= prediction_date)
        ]
        quantiles_subset = quantiles.resample("W-Sat", on="date").sum().reset_index()

        ax.fill_between(
            quantiles_subset["date"],
            quantiles_subset[0.16],
            quantiles_subset[0.84],
            color="b",
            alpha=0.2,
            label="68% CI",
        )

        ax.fill_between(
            quantiles_subset["date"],
            quantiles_subset[0.025],
            quantiles_subset[0.975],
            color="b",
            alpha=0.1,
            label="95% CI",
        )
        ax.plot(
            quantiles_subset["date"],
            quantiles_subset[0.5],
            color="b",
            label="Median",
            marker="*",
        )

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Hospitalizations (Weekly Sum)")
    plt.title(
        f'Actual vs Predicted Hospitalizations | Location {location} | Forecast Date: {prediction_date.strftime("%Y-%m-%d")}'
    )
    plt.legend(loc="upper right")

    plt.tight_layout()

    if streamlit:
        st.pyplot(plt)  # For rendering in Streamlit
    else:
        plt.show()


def plot_beta_forecast(
    beta_data_file_path: str, ensemble_file_path: str, streamlit: bool = False
):
    """
    Plots the beta forecast results.
    """
    date_str = os.path.basename(os.path.dirname(beta_data_file_path))
    loc_str = os.path.splitext(os.path.basename(beta_data_file_path))[0]

    beta_df = pd.read_csv(beta_data_file_path)
    ensemble_df = pd.read_csv(ensemble_file_path)

    # Renaming columns for consistency
    beta_df["time_1"] = beta_df["time_0"] + 1
    beta_df.rename(columns={"time_1": "Day", "beta": "Beta"}, inplace=True)

    combined_df = pd.concat([beta_df, ensemble_df], ignore_index=True, sort=False)

    # Get the last beta value
    last_beta_day = beta_df["Day"].max()
    last_beta_value = beta_df.loc[beta_df["Day"] == last_beta_day, "Beta"].values[0]

    # Add the last beta value as the first point in ensemble_df for a smooth connection
    first_forecast_day = last_beta_day + 1
    ensemble_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Day": [last_beta_day],
                    "Mean": [last_beta_value],
                    "Lower": [last_beta_value],
                    "Upper": [last_beta_value],
                }
            ),
            ensemble_df,
        ],
        ignore_index=True,
    )

    # Plotting
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x="Day", y="Beta", data=beta_df, label="PMCMC Beta Estimation", color="blue"
    )
    sns.lineplot(
        x="Day",
        y="Mean",
        data=ensemble_df,
        label="Ensemble Mean Forecast",
        color="orange",
    )
    plt.fill_between(
        ensemble_df["Day"],
        ensemble_df["Lower"],
        ensemble_df["Upper"],
        color="gray",
        alpha=0.3,
        label="95% Prediction Interval",
    )

    # Add vertical dotted line at the prediction date
    prediction_day = beta_df["Day"].max()
    plt.axvline(
        x=prediction_day,
        color="black",
        linestyle="--",
        label=f"Prediction Start: {date_str}",
    )

    plt.title(f"Beta Ensemble Forecast for Loc {loc_str}: {date_str}")
    plt.xlabel("Day")
    plt.ylabel("Beta Value")
    plt.legend()

    if streamlit:
        st.pyplot(plt)  # For rendering in Streamlit
    else:
        plt.show()
