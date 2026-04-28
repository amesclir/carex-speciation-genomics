#!/bin/bash
#SBATCH --job-name=install_r_packages
#SBATCH --output=install_r_packages.out
#SBATCH --error=install_r_packages.err
#SBATCH --time=00:30:00
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load conda
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Error: Conda initialization script not found."
    exit 1
fi

# Activate the smudgeplot conda environment
conda activate smudgeplot

# Install remotes if not already installed
R -e 'if(!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")'

# Install argparse
R -e 'install.packages("argparse", repos="https://cloud.r-project.org")'

# Install smudgeplot from GitHub using remotes
R -e 'remotes::install_github("PauloMoraMartinez/Smudgeplot")'

echo "Successfully attempted to install argparse and smudgeplot (from GitHub using remotes) R packages."
