#!/bin/bash
#SBATCH --job-name=fst_syri_plot
#SBATCH --output=fst_syri_plot_%j.out
#SBATCH --error=fst_syri_plot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --time=05:00:00

set -e

# --- Configuración del entorno ---
source ~/miniforge3/etc/profile.d/conda.sh
# Activa el entorno 'base' que es el que estás usando en tu terminal
conda activate plot_env

# --- Ejecución ---
# Asegurarnos de que estamos en el directorio correcto
cd /home/aescudero/wgg/fst_results

echo "Iniciando el script de Python..."
python plot_fst_syri5_test.py
echo "Script finalizado."
