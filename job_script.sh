#!/bin/bash

#SBATCH --job-name=pf-flu-prediction-test
#SBATCH --output=/scratch/apa235/test_output.txt
#SBATCH --nodes=1
#SBATCH --mincpus=4
#SBATCH --time=1:00:00
#SBATCH --chdir=/projects/math_cheny/filter_forecast/

# added echo statements for debugging

srun echo "Starting up..."
srun pwd
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
echo "Installed Python packages"

module load R/4.2.3
export PATH=$PATH:/home/yc424/R/4.2.3/
echo "Loaded R"

python3 cluster_single_loc_test.py
echo "Completed job."