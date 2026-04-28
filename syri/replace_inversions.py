import sys

def replace_compound_inversions(input_file, output_file):
    """
    Normalizes coordinates and structural variation (SV) types for all inversion entries
    to satisfy plotsr's strict requirements.

    1. Reorders B-genome coordinates (Columns 7 & 8) to be ascending (start <= end).
    2. Replaces INVTR, INVDP, and INVTRAL types with the simpler INV in Column 11.
    """
    # Types that need coordinate normalization AND type replacement
    inversion_types = ['INV', 'INVTR', 'INVDP', 'INVTRAL']
    
    # Indices (0-based) for the columns we need to work with
    # IMPORTANT: The correct indices for a standard 12-column SYRI output:
    b_start_index = 6   # Column 7: B-genome start
    b_end_index = 7     # Column 8: B-genome end
    sv_type_index = 10  # Column 11: SV Type
    
    processed_count = 0      # Counts coordinate swaps
    type_replaced_count = 0  # Counts type changes (e.g., INVTR -> INV)
    
    print(f"Starting file processing...")
    print(f"Reading from: {input_file}")
    print(f"Writing corrected data to: {output_file}")
    
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Keep comment lines (starting with #) and headers intact
                if line.startswith('#'):
                    outfile.write(line)
                    continue

                # Use tab ('\t') as the standard SYRI delimiter
                parts = line.strip().split('\t')
                
                # Check if the line has enough columns
                if len(parts) > sv_type_index:
                    sv_type = parts[sv_type_index]
                    
                    # 1. Check if it's an inversion-related entry
                    if sv_type in inversion_types:
                        
                        # --- Coordinate Normalization (Fixing the plotsr crash) ---
                        try:
                            # Parse B-genome coordinates
                            b_start = int(parts[b_start_index])
                            b_end = int(parts[b_end_index])
                            
                            # If the end is less than the start, swap them
                            if b_start > b_end:
                                parts[b_start_index] = str(b_end)
                                parts[b_end_index] = str(b_start)
                                processed_count += 1
                            
                            # --- Type Normalization (Replacing compound inversions) ---
                            # Change all inversion-related types to the simple 'INV'
                            if sv_type != 'INV':
                                parts[sv_type_index] = 'INV'
                                type_replaced_count += 1
                            
                        except ValueError:
                            # Skip lines where coordinates are not numbers
                            pass 

                    # Write the modified or original line back out, joined by tabs
                    outfile.write('\t'.join(parts) + '\n')
                else:
                    # Write lines that don't match the expected format as-is
                    outfile.write(line)
        
        print(f"\nProcessing complete.")
        print(f"Total B-genome coordinates reordered: {processed_count}")
        print(f"Total SV types replaced with 'INV': {type_replaced_count}")

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    input_filename = "syri_final2.out"
    output_filename = "syri_fixed_inversions.out"
    
    replace_compound_inversions(input_filename, output_filename)
```eof

---

## Next Step

Please run the script now:

```bash
python replace_inversions.py
