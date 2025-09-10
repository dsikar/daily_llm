#!/bin/bash

#===============================================================================
# CHECK LIST
#===============================================================================

## 1. Set the working directory
## 2. Set the job name
## 3. Set the email type and address
## 4. Set the partition (gengpu or preemptgpu)
## 5. Set the number of nodes
## 6. Set the number of tasks per node
## 7. Set the number of CPUs per task
## 8. Set the expected CPU RAM needed
## 9. Set the time limit
## 10. Set the GPU configuration
## 11. Set the output configuration
## 12. Set the environment setup
## 13. Set the main script
## 14. Set the email job output and calculate duration

#===============================================================================
# SLURM Batch Script Template with Email Notification
#===============================================================================

#SBATCH -D /users/aczd097/git/daily_llm  # Working directory
#SBATCH --job-name gemma29b # Job name 8 characters or less
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
start=$(date +%s)

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Your commands here
echo "Job started at $(date)"
# python your_script.py --arg1 value1 --arg2 value2

#python gemma29b-loop.py # gemma29b.py
python gemma29b.py
#===============================================================================
# Email Job Output and Calculate Duration
#===============================================================================

# Get the output file path
output_file="outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o"

# Wait for file to be written
sleep 5

# Send last 100 lines by email
tail -n 100 "$output_file" | mail -s "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) Output" daniel.sikar@city.ac.uk

# Calculate execution time
end=$(date +%s)
diff=$((end-start))
hours=$((diff / 3600))
minutes=$(( (diff % 3600) / 60 ))
seconds=$((diff % 60))

echo "Job completed at $(date)"
echo "Total execution time: $hours hours, $minutes minutes, $seconds seconds"
