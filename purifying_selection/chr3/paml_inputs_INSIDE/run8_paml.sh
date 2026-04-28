#!/bin/bash

# Configuration: We assume codeml is in your PATH.
CODEML_EXECUTABLE="codeml"
INPUT_DIR="." # Since you are running this from inside paml_inputs/

echo "--- Starting PAML Batch Run ---"
echo "Fixing EOF newline issue for all .phy files..."
# Run the newline fix again just to be safe
sed -i '$a\' *.phy

# Loop through all control files (*.ctl) in the current directory
for ctl_file in "$INPUT_DIR"/*.ctl; do
    # Extract the file name (without path) for logging
    base_name=$(basename "$ctl_file")
    
    # Define a log file for this specific run
    log_file="${base_name%.ctl}.log"
    
    echo "Processing $base_name -> Output logged to $log_file"
    
    # Run codeml, redirecting stdout and stderr to the log file.
    # This prevents the terminal mixing that caused the previous error.
    "$CODEML_EXECUTABLE" "$ctl_file" > "$log_file" 2>&1

    # Check the exit status of the codeml run
    if [ $? -eq 0 ]; then
        echo "  [SUCCESS] $base_name completed."
    else
        echo "  [FAILURE] $base_name failed. Check $log_file for details."
    fi

done

echo "--- PAML Batch Run Complete ---"

