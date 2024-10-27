"""Contains global environment variables."""

import os

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
JOB_SCRIPTS_DIR = os.path.join(ROOT_DIR, 'job_scripts')
UTILS_DIR = os.path.join(SRC_DIR, "utils")
PMCMC_DIR = os.path.join(SRC_DIR, "pmcmc")
TREND_FORECAST_DIR = os.path.join(SRC_DIR, "trend_forecast")
HOSP_FORECAST_DIR = os.path.join(SRC_DIR, "hosp_forecast")
FLASK_DIR = os.path.join(SRC_DIR, "flask_dashboard")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
PF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pf_avg_betas")
TREND_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "trend_forecast")
HOSP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "hosp_forecast")
PMCMC_RUNS_DIR = os.path.join(OUTPUT_DIR, "pmcmc_runs")

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

# This SPHERE is only intended for testing on Andrew's local machine.
SPHERE_DIR = "/home/andrew/PycharmProjects/SPHERE/sphere"
