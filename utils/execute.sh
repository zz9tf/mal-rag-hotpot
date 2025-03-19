#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=prepro_xml
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=./out/execute.out
#SBATCH --cpus-per-task=100

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python wikipedia_database_preprocessor.py --action parse