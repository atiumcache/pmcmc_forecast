#!/bin/bash
#SBATCH --job-name=beta_mu
#SBATCH --output=beta_mu_%A_%a.out # Output file (%A = job ID, %a = array task ID)
#SBATCH --error=beta_mu_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-2                  # Array job index (3 values: 0, 1, 2)

# Define the list of target dates (one per array task)
target_dates=("2023-12-01" "2024-01-30" "2023-04-20")

echo -e "\nJob started at: $(date)\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
python3 -m pip uninstall numpy
python3 -m pip install --upgrade numexpr bottleneck
python3 -m pip install numpy<2
echo -e "\nInstalled Python packages\n"

# Run the Python script, passing the corresponding target date based on SLURM_ARRAY_TASK_ID
python3 -m src.scripts.202401023_beta_mu ${target_dates[$SLURM_ARRAY_TASK_ID]}

echo -e "\nPython script completed at: $(date)\n"
echo -e "\n Completed job.\n"
