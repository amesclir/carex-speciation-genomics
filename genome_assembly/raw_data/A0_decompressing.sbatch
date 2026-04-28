#!/usr/bin/bash
#SBATCH --job-name=decompressing_multiple
#SBATCH --output=decompressing_multiple_output
#SBATCH --error=decompressing_multiple_error
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=40G
#SBATCH --partition=standard
#SBATCH --time=12:00:00

RAW_DATA_DIR="/home/aescudero/genome_assembly/raw_data"
SAMPLES_FILE="$RAW_DATA_DIR/samples.txt"
HIFI_DIR="PacBio"
HIC_DIR="hiC"

# Check if the samples file exists
if [ ! -f "$SAMPLES_FILE" ]; then
  echo "Error: The samples file '$SAMPLES_FILE' does not exist."
  exit 1
fi

# Process each sample listed in the samples file
while IFS= read -r sample; do
  echo "Processing sample: $sample"
  echo "-------------------------"

  # --- Decompress and concatenate HiFi data (if present) ---
  HIFI_SAMPLE_DIR="$RAW_DATA_DIR/$HIFI_DIR/$sample"
  if [ -d "$HIFI_SAMPLE_DIR" ]; then
    echo "Decompressing and concatenating HiFi files in: $HIFI_SAMPLE_DIR"
    cd "$HIFI_SAMPLE_DIR" || { echo "Error changing directory to $HIFI_SAMPLE_DIR"; continue; }
    # Find all .fastq.gz files, decompress them, and concatenate
    find . -maxdepth 1 -name "*.fastq.gz" -print0 | while IFS= read -r -d $'\0' file; do
      gunzip -c "$file" >> "${sample}_hifi.fastq"
      echo "Decompressed and added to ${sample}_hifi.fastq: $file"
    done
    if [ -s "${sample}_hifi.fastq" ]; then
      echo "Successfully created concatenated HiFi file: ${sample}_hifi.fastq"
    else
      echo "Warning: No HiFi data was found and/or concatenated for $sample."
    fi
    cd - > /dev/null # Go back to the previous directory
  else
    echo "Warning: HiFi directory '$HIFI_SAMPLE_DIR' does not exist."
  fi

  # --- Decompress Hi-C data ---
  HIC_SAMPLE_DIR="$RAW_DATA_DIR/$HIC_DIR/$sample"
  if [ -d "$HIC_SAMPLE_DIR" ]; then
    echo "Decompressing Hi-C files in: $HIC_SAMPLE_DIR"
    cd "$HIC_SAMPLE_DIR" || { echo "Error changing directory to $HIC_SAMPLE_DIR"; continue; }
    # Decompress all .fastq.gz files, keeping their original names (without .gz)
    find . -maxdepth 1 -name "*.fastq.gz" -print0 | while IFS= read -r -d $'\0' file; do
      base_name=$(basename "$file" .gz)
      gunzip "$file"
      echo "Decompressed: $file -> $base_name"
    done
    cd - > /dev/null # Go back to the previous directory
  else
    echo "Warning: Hi-C directory '$HIC_SAMPLE_DIR' does not exist."
  fi

  echo "Finished processing sample: $sample"
done < "$SAMPLES_FILE"

echo "Decompression and concatenation of all samples complete."
