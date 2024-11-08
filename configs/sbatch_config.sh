#!/bin/bash
#SBATCH --job-name=fairness_embedding
#SBATCH --output=fairness_embedding_%j.out
#SBATCH --error=fairness_embedding_%j.err
#SBATCH --gres=gpu:4                # Request 4 GPUs
#SBATCH --ntasks=4                  # Run 4 tasks
#SBATCH --cpus-per-task=25          # 100 CPUs divided among 4 tasks
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00

# Change to the project directory
cd /home/hhamidi/clean/projects/fairness_on_embeddings

# Activate the conda environment
source activate bm

# Array of configuration files
CONFIGS=(
  "mimic_base_NoFinding.yaml"
  "mimic_base_Cardiomegaly.yaml"
  "mimic_base_Consolidation.yaml"
  "mimic_base_Edema.yaml"
  "mimic_base_Effusion.yaml"
)

# Run the first four tasks concurrently, each on a separate GPU
for i in {0..3}; do
  srun --exclusive --gres=gpu:1 --ntasks=1 --cpus-per-task=25 \
  python main.py fit -c ./configs/${CONFIGS[$i]} &
done

wait  # Wait for the first four tasks to complete

# Run the fifth task
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=25 \
python main.py fit -c ./configs/${CONFIGS[4]}
