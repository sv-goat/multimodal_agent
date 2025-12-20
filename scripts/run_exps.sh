#!/bin/bash
#SBATCH -A edu
#SBATCH --job-name=qwen_experiments
#SBATCH --output=logs/exp_%j.out   # Standard output log
#SBATCH --error=logs/exp_%j.err    # Standard error log
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --time=12:00:00             # Time limit
#SBATCH --partition=short

# --- 1. Configuration ---
# Your large writable storage path
BASE_DIR="/insomnia001/depts/edu/COMS-E6998-015/sv2795"

# Where your container lives
SIF_IMAGE="/insomnia001/home/sv2795/tyrani.sif"

# Your python master script (the one that loops over parameters)
PYTHON_SCRIPT="experiment/master_runner.py"

# --- 2. Directory Setup ---
# Create directories on the host so the container can map them
echo "Setting up directories..."
mkdir -p "$BASE_DIR/wandb_data"

# --- 3. Run Apptainer ---
echo "Starting Container..."

# We use 'exec' instead of 'shell' to run a specific command and exit.
# We pass environment variables directly into the container context.

apptainer exec --nv --writable-tmpfs \
  -B "$BASE_DIR:/data" \
  --env HF_HOME="/data/hf_cache" \
  --env WANDB_DIR="/data/wandb_data" \
  --env WANDB_CACHE_DIR="/data/wandb_data/cache" \
  --env WANDB_CONFIG_DIR="/data/wandb_data/config" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  "$SIF_IMAGE" \
  /bin/bash -c "
    # --- INSIDE CONTAINER ---
    echo 'Container started.'
    
    # 1. Activate Virtual Env (If your venv is INSIDE the container, skip this.
    #    If your venv is a folder on the host, map it and source it here.)
    source /insomnia001/home/sv2795/ocr/bin/activate
    
    # 2. Verify GPU visibility
    nvidia-smi

    # Stop torch compile
    export VLLM_TORCH_COMPILE_LEVEL=0;  # Tell vLLM not to compile
    export TORCH_COMPILE_DISABLE=1;      # Tell PyTorch to disable compile
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;
    
    # 3. Run your Master Python Script
    # We assume the script is in the current folder where you submit the job
    echo 'Running experiments...'
    python3 $PYTHON_SCRIPT
  "

echo "Job finished."
