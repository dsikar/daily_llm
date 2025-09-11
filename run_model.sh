#!/bin/bash

# Usage: ./run_model.sh [job_name] [python_script]
# Defaults: job_name=runmodel, python_script=run_model.py

# Check if this is being called as a submission wrapper
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] && [[ ! -v SLURM_JOB_ID ]]; then
    # This is the wrapper - create and submit the actual SLURM script
    JOB_NAME=${1:-runmodel}
    PYTHON_SCRIPT=${2:-run_model.py}
    
    # Create temporary SLURM script
    TEMP_SCRIPT=$(mktemp /tmp/slurm_job_XXXXXX.sh)
    
    # Generate the SLURM script with substituted variables
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash

#===============================================================================
# SLURM Batch Script Template with Email Notification
#===============================================================================

#SBATCH -D /users/aczd097/git/daily_llm  # Working directory
#SBATCH --job-name ${JOB_NAME} # Job name 8 characters or less
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk     # Where to send mail

#===============================================================================
# Resource Configuration
#===============================================================================

#SBATCH --partition=preemptgpu              # Partition choice gengpu or preemptgpu 
##SBATCH --partition=nodes              # Run on nodes partition
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=4               # CPUs per task
#SBATCH --mem=30GB                       # Expected CPU RAM needed, 0 use all
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --array=1-10                    # Run this script 10 times

#===============================================================================
# GPU Configuration
#===============================================================================

# Choose one of these GPU configurations:
##SBATCH --gres=gpu:a100:1              # Request 1x A100 40GB GPU gengpu partition
#SBATCH --gres=gpu:a100_80g:1         # Request 1x A100 80GB GPU premptgpu partition

#===============================================================================
# Output Configuration
#===============================================================================
#SBATCH -e outputs/%x_%j.e # Standard error log
#SBATCH -o outputs/%x_%j.o # Standard output log
# %j = job ID, %x = job name
#===============================================================================
# Environment Setup
#===============================================================================

# Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

# Clean environment
module purge

# Load required modules
module add gnu
# Add other required modules here

#===============================================================================
# Main Script
#===============================================================================

# Record start time
start=\$(date +%s)

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Your commands here
echo "Job started at \$(date)"
echo "Array Task ID: \${SLURM_ARRAY_TASK_ID}"

# Run the model inference script
python ${PYTHON_SCRIPT}

#===============================================================================
# Email Job Output and Calculate Duration
#===============================================================================

# Get the output file path
output_file="outputs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}.o"

# Wait for file to be written
sleep 5

# Send last 100 lines by email
tail -n 100 "\$output_file" | mail -s "Job \${SLURM_JOB_NAME} (\${SLURM_JOB_ID}) Output" daniel.sikar@city.ac.uk

# Calculate execution time
end=\$(date +%s)
diff=\$((end-start))
hours=\$((diff / 3600))
minutes=\$(( (diff % 3600) / 60 ))
seconds=\$((diff % 60))

echo "Job completed at \$(date)"
echo "Total execution time: \$hours hours, \$minutes minutes, \$seconds seconds"
EOF

    # Submit the generated script
    echo "Submitting job with name: $JOB_NAME, script: $PYTHON_SCRIPT"
    sbatch "$TEMP_SCRIPT"
    
    # Clean up
    rm "$TEMP_SCRIPT"
    
    exit 0
fi