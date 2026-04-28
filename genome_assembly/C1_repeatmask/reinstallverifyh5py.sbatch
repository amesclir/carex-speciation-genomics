#!/bin/bash
#SBATCH --job-name=reinstall_h5py
#SBATCH --output=reinstall_h5py_output
#SBATCH --error=reinstall_h5py_error
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 # Only need 1 CPU for these operations
#SBATCH --mem=4G # Should be enough memory for these operations
#SBATCH --partition=standard
#SBATCH --time=01:00:00 # 1 hour should be plenty for installation and checks

# Set source for conda
# This command should correctly locate and source your base conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Starting h5py reinstallation and verification checks..."

# Activate the repeatmasker environment
# The || { ...; } construct will exit the script if the environment activation fails
conda activate repeatmasker || { echo "Failed to activate repeatmasker conda environment! Is it installed correctly?"; exit 1; }
echo "Conda environment 'repeatmasker' activated."

# --- Step 1: Attempting to reinstall h5py ---
echo "--- Step 1: Attempting to reinstall h5py ---"
# Using --force-reinstall to ensure it's reinstalled even if already present
# Using -y for non-interactive mode (answers yes to prompts)
conda install h5py --force-reinstall -y
if [ $? -eq 0 ]; then # Check the exit status of the previous command
    echo "h5py reinstallation command completed successfully."
else
    echo "h5py reinstallation command FAILED! Please check the 'reinstall_h5py_error' file for details."
    conda deactivate # Deactivate environment before exiting
    exit 1 # Exit the script with an error code
fi
echo "------------------------------------------"

# --- Checkpoint 1: Verify h5py installation ---
echo "--- Checkpoint 1: h5py version after reinstallation ---"
conda list h5py # Lists installed packages, including h5py version
echo "----------------------------------------------------"

# --- Checkpoint 2: Test h5py import via Python ---
echo "--- Checkpoint 2: Testing h5py import in Python ---"
# This Python command tries to import h5py; errors are redirected to a log file
python -c "import h5py; print('h5py imported successfully!');" > h5py_import_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "h5py import test PASSED."
    cat h5py_import_test.log # Display the log content
else
    echo "h5py import test FAILED. Please check 'h5py_import_test.log' for details."
    cat h5py_import_test.log
    conda deactivate
    exit 1 # Exit if h5py still cannot be imported
fi
echo "-------------------------------------------------"

# --- Checkpoint 3: Test famdb.py execution (which uses h5py) ---
# This path is crucial; ensure it's correct for your RepeatMasker installation
FAMDB_SCRIPT="/lustre/home/aescudero/anaconda3/envs/repeatmasker/share/RepeatMasker/famdb.py"
echo "--- Checkpoint 3: Testing famdb.py execution ---"
if [ ! -f "$FAMDB_SCRIPT" ]; then # Check if the famdb.py script exists
    echo "Error: famdb.py script not found at '$FAMDB_SCRIPT'. Please verify the path."
    conda deactivate
    exit 1
fi
# Execute famdb.py to trigger its h5py dependency
python "$FAMDB_SCRIPT" info > famdb_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "famdb.py execution test PASSED."
    cat famdb_test.log
else
    echo "famdb.py execution test FAILED. Please check 'famdb_test.log' for details."
    cat famdb_test.log
    conda deactivate
    exit 1 # Exit if famdb.py still fails
fi
echo "------------------------------------------------"

# --- Checkpoint 4: Run a minimal RepeatMasker command to verify overall functionality ---
echo "--- Checkpoint 4: Minimal RepeatMasker functionality test ---"
# Create a dummy fasta file for a quick test run
echo ">test_sequence" > test_rm_input.fasta
echo "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC" >> test_rm_input.fasta

# Run RepeatMasker using -species "plants" (this often triggers famdb.py internally)
# -pa 1 for single-threaded quick test, -dir . to put outputs in current directory
RepeatMasker -pa 1 -e rmblast -species "plants" -dir . -xsmall test_rm_input.fasta > rm_minimal_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "Minimal RepeatMasker run test PASSED."
    # Check if the output contains a common success indicator for masking
    grep -q "masking" rm_minimal_test.log && echo "Confirmation: 'masking' message found in test log." || echo "Warning: 'masking' message not found in test log, but command exited successfully."
else
    echo "Minimal RepeatMasker run test FAILED. Please check 'rm_minimal_test.log' for details."
    cat rm_minimal_test.log
    conda deactivate
    exit 1 # Exit if RepeatMasker itself still fails its basic function
fi

# Clean up dummy test files created by this script
rm -f test_rm_input.fasta test_rm_input.fasta.out test_rm_input.fasta.tbl test_rm_input.fasta.html test_rm_input.fasta.gff rm_minimal_test.log h5py_import_test.log famdb_test.log

echo "---------------------------------------------------"

# Deactivate conda environment
conda deactivate
echo "Conda environment deactivated."
echo "All h5py reinstallation and verification checks completed. Please check all output files for success messages."
