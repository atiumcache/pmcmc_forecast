#!/bin/bash

#SBATCH --job-name=trend_forecast_test
#SBATCH --output=/scratch/apa235/trend_forecast_test.txt
#SBATCH --nodes=1
#SBATCH --mincpus=64
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/math_cheny/pmcmc_forecast/
#SBATCH --mem=32GB

# added echo statements for debugging

echo -e "\nJob started at: $(date)\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
echo -e "\nInstalled Python packages\n"

echo -e "\nRunning the Python script... \n"
echo -e "\nPython script started at: $(date)\n"
python3 -m src.trend_forecast.main

echo -e "\nPython script completed at: $(date)\n"
echo -e "\n Completed job.\n"
