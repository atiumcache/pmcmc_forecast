#!/bin/bash

# Set the base directory
# This assumes that this script is being run
# one level down from root dir.
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
JOB_SCRIPTS_DIR="${BASE_DIR}/job_scripts"

# Check if a date argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 YYYY-MM-DD"
    exit 1
fi

DATE="$1"

# Construct the full path to the date-specific directory in job_scripts
DATE_DIR="${JOB_SCRIPTS_DIR}/${DATE}"

# Check if the date directory exists
if [ ! -d "$DATE_DIR" ]; then
    echo "Error: Job scripts directory $DATE_DIR does not exist."
    exit 1
fi

# Set PYTHONPATH to include the root directory
export PYTHONPATH="${BASE_DIR}:$PYTHONPATH"

# Loop through each .sh file in the date directory and submit it
for script in "$DATE_DIR"/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done

echo "All jobs for $DATE have been submitted."
