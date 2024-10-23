import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils import paths, plotting


location_csv_path = os.path.join(paths.DATASETS_DIR, "locations.csv")
location_df = pd.read_csv(location_csv_path)
location_df = location_df.query("location_name != 'US'")
state_to_loc = dict(zip(location_df["location_name"], location_df["location"]))

target_dates = pd.read_csv(os.path.join(paths.DATASETS_DIR, "target_dates.csv"))
dates = target_dates["date"].values


st.title("Forecasting Dashboard")
# Example dropdown to select state and date

# Example time series data for the selected state (replace with your data)
time = np.arange(0, 10, 0.1)
values = np.sin(time) + np.random.randn(len(time)) * 0.1  # Example values

# User selects state
selected_state = st.selectbox("Select a State", list(state_to_loc.keys()))

# Get corresponding loc_code
loc_code = state_to_loc[selected_state]

# Call the function with loc_code and user input
prediction_date = st.selectbox("Select Forecast Date", list(dates))
if st.button("Plot Hospitalizations"):
    try:
        plotting.plot_predictions_with_quantile_range(
            str(prediction_date),
            loc_code,
            pf_uncertainty=True,
            streamlit=True,
            hosp_est_file_name="mle_hosp_est_20241020.npy",
            weeks_prior=8,
            daily_resolution=False
        )
    except FileNotFoundError:
        st.write(
            "File not found. This forecast has probably not been run yet. Try another location/date."
        )


# Time series section
# USA heatmap section
# st.subheader('USA Heatmap of Flu Hospitalizations')
# st.plotly_chart(fig1)
