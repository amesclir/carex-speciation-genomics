#!/bin/bash
#SBATCH --job-name=Check_HiC_Software
#SBATCH --output=check_hic_software_output.txt
#SBATCH --error=check_hic_software_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G          # Minimal memory needed for just checking tools
#SBATCH --time=00:05:00   # Short time limit

echo "--- Starting Hi-C Software Check Script ---"
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo ""

# --- Conda Environment Activation ---
echo "Attempting to source Conda and activate 'ref_map' environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ref_map

# Check if conda activation was successful
if [ $? -eq 0 ]; then
    echo "Conda environment 'ref_map' activated successfully."
    echo "Current Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "ERROR: Failed to activate Conda environment 'ref_map'. Check check_hic_software_error.txt for details."
    echo "This might indicate that the environment doesn't exist, is corrupted, or there's still an underlying permission issue."
    exit 1 # Exit if activation fails, as subsequent checks will be pointless
fi
echo ""

# --- Set and Verify JAVA_HOME for Picard ---
# This assumes your 'ref_map' environment (or system) provides Java, or you have it installed.
# It tries to find the JAVA_HOME based on where 'java' is found in PATH.
echo "Verifying Java installation and setting JAVA_HOME..."
if which java >/dev/null 2>&1; then
    echo "Java executable found at: $(which java)"
    export JAVA_HOME=$(dirname $(dirname $(which java)))
    echo "Set JAVA_HOME to: $JAVA_HOME"
    if [ -d "$JAVA_HOME" ]; then
        echo "JAVA_HOME directory exists: $JAVA_HOME."
        echo "Java version:"
        java -version 2>&1 | grep -E "openjdk version|java version"
    else
        echo "WARNING: JAVA_HOME directory does NOT exist: $JAVA_HOME. This might be an issue for Picard."
    fi
else
    echo "ERROR: Java command not found. Picard will likely fail without Java."
fi
echo ""

# --- Check presence and version of specific tools ---
echo "Checking for essential bioinformatics tools and their versions..."

# Define tools and commands to check their versions
# For Picard, use 'picard --version' if it's a wrapper script,
# or 'java -jar /path/to/picard.jar --version' if it's just the JAR.
# The `which picard` command should ideally point to a wrapper in your Conda bin.
declare -A TOOLS_CHECK
TOOLS_CHECK["bwa"]="bwa version"
TOOLS_CHECK["fastp"]="fastp --version"
TOOLS_CHECK["samtools"]="samtools --version"
TOOLS_CHECK["picard"]="picard --version" # Assumes a 'picard' wrapper executable is in PATH

for tool in "${!TOOLS_CHECK[@]}"; do
    echo -n "Checking '$tool': "
    if which "$tool" >/dev/null 2>&1; then
        echo "FOUND at $(which "$tool")"
        echo "  Version Info:"
        # Run the specific version check command for each tool
        if [ "$tool" == "fastp" ] || [ "$tool" == "samtools" ]; then
            # fastp and samtools print version to stderr by default
            "${TOOLS_CHECK[$tool]}" 2>&1 | head -n 1 | sed 's/^Version: //'
        elif [ "$tool" == "picard" ]; then
            # Picard's --version output can be multi-line or specific
            "${TOOLS_CHECK[$tool]}" 2>&1 | grep -E "Picard|version"
        else
            # For other tools, just run the command and get first line
            "${TOOLS_CHECK[$tool]}" 2>&1 | head -n 1
        fi
    else
        echo "NOT FOUND"
        echo "  --> '$tool' is not in your PATH after 'conda activate ref_map'."
        echo "  --> Please ensure it is installed in your 'ref_map' environment (e.g., 'conda install -c bioconda $tool')."
    fi
    echo ""
done

echo "--- Software check complete ---"
echo "Job finished on: $(date)"
