"""
Generate shell scripts for each location,
given a forecast date.

This script expects the target/forecast date
as the first (and only) command line arg.

Usage:
    python3 src.scripts.generate_shell_scripts 'YYYY-MM-DD'
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


from src.utils import paths


target_date = sys.argv[1]

# Define base directories
slurm_logs_dir = os.path.join(paths.OUTPUT_DIR, 'slurm_logs', target_date)
os.makedirs(slurm_logs_dir, exist_ok=True)

# Read the CSV file
location_csv_path = os.path.join(paths.DATASETS_DIR, 'locations.csv')
with open(location_csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Template for the SLURM script
slurm_template = """
#!/bin/bash
#SBATCH --job-name=forecast_{location}
#SBATCH --output={slurm_logs_dir}/output_{location}_{date}.out
#SBATCH --error={slurm_logs_dir}/error_{location}_{date}.err

# Location and date from parameters
LOCATION="{location}"
DATE="{date}"

echo "Starting forecast for location: $LOCATION. Forecast date: $DATE"

python3 -m src.scripts.flusight_forcecast $LOCATION $DATE
"""

# Generate a SLURM script for each row in the CSV
for row in rows:
    location = row['location']
    slurm_script = slurm_template.format(location=location,
                                         date=target_date,
                                         slurm_logs_dir=slurm_logs_dir)
    
    # Define the filename based on location and date
    script_filename = f"job_{location}_{target_date}.sh"
    script_dir = os.path.join(paths.JOB_SCRIPTS_DIR, target_date)
    script_path = os.path.join(script_dir, script_filename)
    os.makedirs(script_dir, exist_ok=True)
    
    # Write the SLURM script to a file
    with open(script_path, 'w') as script_file:
        script_file.write(slurm_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)

print("SLURM job scripts generated successfully!")
