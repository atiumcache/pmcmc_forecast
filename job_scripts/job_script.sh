#!/bin/bash

#SBATCH --job-name=pmcmc_beta_generate
#SBATCH --output=/scratch/apa235/.txt
#SBATCH --nodes=1
#SBATCH --mincpus=8
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/math_cheny/pmcmc_forecast/
#SBATCH --mem=32GB

# added echo statements for debugging

echo -e "Starting up...\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
echo -e "\n Installed Python packages\n"


echo -e "\n Running the Python script... \n"
python3 src.pmcmc.main
echo -e "\n Completed job.\n"
