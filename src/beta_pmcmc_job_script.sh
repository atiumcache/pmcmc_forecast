#!/bin/bash

#SBATCH --job-name=pmcmc_beta_generate
#SBATCH --output=/scratch/apa235/diff_coeff_test.txt
#SBATCH --nodes=1
#SBATCH --mincpus=28
#SBATCH --time=36:00:00
#SBATCH --chdir=/projects/math_cheny/pmcmc_tests_andrew/diff_coeff_test/
#SBATCH --mem=32GB

# added echo statements for debugging

echo -e "Starting up...\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
# Was having trouble on Monsoon with numpy/dependencies. Quick fix.
python3 -m pip uninstall numpy
python3 -m pip install --upgrade numexpr bottleneck
python3 -m pip install numpy<2
echo -e "\n Installed Python packages\n"

echo -e "\n Running the Python script... \n"
python3 -m src.pmcmc.main
echo -e "\n Completed job.\n"
