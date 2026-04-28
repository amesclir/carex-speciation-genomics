#!/bin/bash
#SBATCH --job-name=ld_decay_top30
#SBATCH --output=ld_decay_top30_%j.out
#SBATCH --error=ld_decay_top30_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=standard

set -e

echo "--- Starting Targeted LD Decay Workflow ---"

# --- Activate Conda environment ---
CONDA_ENV_NAME="bcftools"
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source "/home/aescudero/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}" || { echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'. Exiting." >&2; exit 1; }
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- Configuration ---
PLINK_FILTERED_PREFIX="/home/aescudero/wgg/RDA/admixture_final_ld_pruned_filtered"
SYRI_FILE="/home/aescudero/syri/syri.out"
RESULTS_DIR="/home/aescudero/wgg/fst_results/windowed_stats"
LD_DIR="${RESULTS_DIR}/ld_decay"
TOP_SVS_LIST="${RESULTS_DIR}/top_30_svs_coordinates.txt"
MAX_DIST_KB="100"

mkdir -p "${LD_DIR}"

echo "Extracting Top 30 SVs from SyRI output..."
awk '{
    if ($11 ~ /^(INV|DUP|TRANS|INVTR|INVDP)$/ && $1 ~ /^scaffold_/) {
        start = $2; end = $3;
        if (start == "-" || end == "-") next;
        
        if (start > end) { temp = start; start = end; end = temp }
        
        size = end - start;
        if (size > 10000) {
            chr = $1; sub("scaffold_", "", chr); 
            print chr, start, end, size, $11
        }
    }
}' "${SYRI_FILE}" | sort -k4,4nr | head -n 30 > "${TOP_SVS_LIST}"

COUNTER=1
while read CHR SV_START SV_END SIZE TYPE; do
    SV_ID="SV${COUNTER}"
    echo "---------------------------------------------------------"
    echo "TARGET ${COUNTER}: ${SV_ID} (${TYPE} on Chr ${CHR}, Size: ${SIZE} bp)"
    
    echo "   - Calculating LD INSIDE ${SV_ID} (Chr ${CHR}: ${SV_START}-${SV_END})..."
    plink --bfile "${PLINK_FILTERED_PREFIX}" \
          --chr-set 34 no-xy \
          --chr ${CHR} \
          --from-bp ${SV_START} \
          --to-bp ${SV_END} \
          --r2 \
          --ld-window-kb ${MAX_DIST_KB} \
          --ld-window 99999 \
          --ld-window-r2 0 \
          --out "${LD_DIR}/${SV_ID}_inside" || echo "Warning: PLINK failed for ${SV_ID}_inside."

    echo "   - Calculating LD OUTSIDE ${SV_ID} (Rest of Chr ${CHR})..."
    
    # 1. Calculate Upstream of the SV
    plink --bfile "${PLINK_FILTERED_PREFIX}" \
          --chr-set 34 no-xy \
          --chr ${CHR} \
          --to-bp ${SV_START} \
          --r2 \
          --ld-window-kb ${MAX_DIST_KB} \
          --ld-window 99999 \
          --ld-window-r2 0 \
          --out "${LD_DIR}/${SV_ID}_outside_up" || echo "Warning: No upstream SNPs."

    # 2. Calculate Downstream of the SV
    plink --bfile "${PLINK_FILTERED_PREFIX}" \
          --chr-set 34 no-xy \
          --chr ${CHR} \
          --from-bp ${SV_END} \
          --r2 \
          --ld-window-kb ${MAX_DIST_KB} \
          --ld-window 99999 \
          --ld-window-r2 0 \
          --out "${LD_DIR}/${SV_ID}_outside_down" || echo "Warning: No downstream SNPs."

    # 3. Merge the Outside Files cleanly
    OUTSIDE_FINAL="${LD_DIR}/${SV_ID}_outside.ld"
    > "${OUTSIDE_FINAL}" # clear file if it exists

    if [ -f "${LD_DIR}/${SV_ID}_outside_up.ld" ]; then
        cat "${LD_DIR}/${SV_ID}_outside_up.ld" > "${OUTSIDE_FINAL}"
    fi

    if [ -f "${LD_DIR}/${SV_ID}_outside_down.ld" ]; then
        if [ -s "${OUTSIDE_FINAL}" ]; then
            tail -n +2 "${LD_DIR}/${SV_ID}_outside_down.ld" >> "${OUTSIDE_FINAL}"
        else
            cat "${LD_DIR}/${SV_ID}_outside_down.ld" > "${OUTSIDE_FINAL}"
        fi
    fi

    rm -f "${LD_DIR}/${SV_ID}_outside_up"* "${LD_DIR}/${SV_ID}_outside_down"*

    COUNTER=$((COUNTER + 1))
done < "${TOP_SVS_LIST}"

echo "---------------------------------------------------------"
echo "LD calculation finished for Top 30 SVs. Outputs saved to ${LD_DIR}"
echo "--- Workflow Completed Successfully ---"
