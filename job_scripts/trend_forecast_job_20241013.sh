#!/bin/bash

#SBATCH --job-name=trend_forecast_06
#SBATCH --output=beta_forecast_06_%j.out
#SBATCH --error=beta_forecast_06_%j.err
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --chdir=/projects/math_cheny/pmcmc_tests_andrew/pmcmc_forecast

echo -e "\nJob started at: $(date)\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
python3 -m pip uninstall numpy
python3 -m pip install --upgrade numexpr bottleneck
python3 -m pip install numpy<2
echo -e "\nInstalled Python packages\n"

module load R/4.2.3
echo -e "\n Loaded R\n"

echo -e "\nRunning the Python script... \n"
echo -e "\nPython script started at: $(date)\n"
python3 -m src.trend_forecast.main

echo -e "\nPython script completed at: $(date)\n"
echo -e "\n Completed job.\n"
