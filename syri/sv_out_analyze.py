# File: sv_out_analyzer_strict.py
# Description: Parses SYRI output using STRICT FILTERING. 
# Only events with a RAW_TYPE EXACTLY matching one of: 
# SYN, INV, TRANS, INVTR, DUP, or INVDP are considered. 
# Events with types like 'INVAL', 'SYNAL', etc., are COMPLETELY ignored.

import sys
import os
from collections import Counter
from typing import List, Tuple, Dict, Counter as TCounter

# --- Configuration ---
# Define the minimum size threshold for a rearrangement to be included
THRESHOLD_BP = 10000
MAX_RESULTS = 30

# The only SV types allowed in the report (must be an EXACT string match)
EXACT_REPORT_TYPES = ['SYN', 'INV', 'TRANS', 'INVTR', 'DUP', 'INVDP']
DISCRETE_SV_TYPES = ['INV', 'TRANS', 'INVTR', 'DUP', 'INVDP'] # Used for the Top 20 list

# The columns used for the syri.out format (0-indexed)
COL_REF_CHR = 0
COL_REF_START = 1
COL_REF_END = 2
COL_SV_TYPE = 10 
MIN_COLUMNS = 11

# Type alias for clarity: (Ref_Chr, Start, End, SV_Type, Size)
SV_Record = Tuple[str, int, int, str, int]

# --- Core Functions ---

def calculate_size_and_type(parts: List[str]) -> Tuple[int, str]:
    """
    Calculates the size from Ref_End - Ref_Start and extracts the SV type.
    """
    try:
        ref_start = int(parts[COL_REF_START])
        ref_end = int(parts[COL_REF_END])
        sv_type = parts[COL_SV_TYPE].upper()
        
        # Calculate size as the absolute difference between coordinates
        size_abs = abs(ref_end - ref_start)
        
        return size_abs, sv_type
    except (ValueError, IndexError):
        # Return 0 size and empty type if parsing fails
        return 0, ""


def read_and_process_sv_file(filepath: str) -> Tuple[List[SV_Record], TCounter]:
    """
    Reads the syri.out file, strictly filters records by size AND type (exact match).
    """
    # This list holds all events >= 10kb AND matching the exact type list
    all_large_events: List[SV_Record] = []
    # This counter stores counts for the 6 EXACT requested types
    exact_sv_counts: TCounter = Counter()
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith(('#', 'Ref_chr')) or not line.strip():
                    continue

                parts = line.strip().split()
                
                if len(parts) < MIN_COLUMNS:
                    continue
                
                size_abs, sv_type = calculate_size_and_type(parts)

                # Ignore non-structural variants and events with size 0
                if sv_type in ('SNP', 'INS', 'DEL') or size_abs == 0:
                    continue
                
                # *** CRITICAL STRICT FILTERING STEP ***
                # 1. Filter: Keep only large events >= the defined threshold
                # 2. Filter: Keep only events where SV_TYPE is an EXACT match
                if size_abs >= THRESHOLD_BP and sv_type in EXACT_REPORT_TYPES:
                    ref_chr = parts[COL_REF_CHR]
                    # Ensure start < end for consistent reporting
                    start = min(int(parts[COL_REF_START]), int(parts[COL_REF_END]))
                    end = max(int(parts[COL_REF_START]), int(parts[COL_REF_END]))
                    
                    record = (ref_chr, start, end, sv_type, size_abs)
                    all_large_events.append(record)
                    exact_sv_counts[sv_type] += 1
                    
        return all_large_events, exact_sv_counts

    except FileNotFoundError:
        print(f"\nERROR: The file path '{filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred while reading the file: {e}")
        sys.exit(1)

def generate_report(all_events: List[SV_Record], exact_counts: TCounter, filepath: str):
    """
    Generates and prints the final report with the exact match summary and Top N list.
    """
    
    total_requested_svs = sum(exact_counts.values())
    
    print("\n" + "=" * 80)
    print(f"       SYRI Rearrangement Report: Filtering $\ge$ {THRESHOLD_BP/1000:,} kb")
    print("=" * 80)
    print(f"Source File: {filepath}")
    
    if total_requested_svs == 0:
        print(f"\nNo large events found with a calculated size of {THRESHOLD_BP:,} bp or greater matching the exact types: {', '.join(EXACT_REPORT_TYPES)}.")
        return

    # --- 1. Exact Match Summary (Totals for the 6 requested categories) ---
    print("\n## Filtered Structural Variant Summary ($\ge$ 10 kb)")
    print(f" (Counts ONLY the EXACT types: {', '.join(EXACT_REPORT_TYPES)})")
    print("-" * 40)
    
    header = f"{'TYPE':<10}{'COUNT':>10}{'PERCENT':>15}"
    print(header)
    print("-" * 40)

    # Print the requested types, sorted by count (including types with 0 count if desired, 
    # but here we just print what was counted)
    if total_requested_svs > 0:
        # Sort by the predefined order (SYN first, then others by count) or just by count
        # For consistency with the user's example, we'll sort by count descending.
        sorted_counts = sorted(exact_counts.items(), key=lambda item: item[1], reverse=True)
        
        # Calculate percentages and print
        for sv_type, count in sorted_counts:
            percent = (count / total_requested_svs) * 100
            # Print count with thousands separator
            print(f"{sv_type:<10}{count:>10,}{percent:>14.1f}%")
    else:
        print(f"No events of the requested EXACT types ($\ge$ {THRESHOLD_BP/1000:,} kb) found.")

    print("-" * 40)
    print(f"TOTAL: {total_requested_svs:,} events of requested types.")
    print("=" * 80 + "\n")

    # --- 2. Top 20 Filtered List (Excludes EXACT 'SYN' matches) ---
    
    # Filter: Keep only events where the type is in the DISCRETE_SV_TYPES list (i.e., not 'SYN')
    requested_rearrangements = [
        record for record in all_events 
        if record[3] in DISCRETE_SV_TYPES
    ]
    
    total_requested = len(requested_rearrangements)
    
    # Sort by absolute size in descending order
    sorted_requested = sorted(requested_rearrangements, key=lambda x: x[4], reverse=True)

    # Select Top N (which is 20) from the sorted list
    top_n_rearrangements = sorted_requested[:MAX_RESULTS]
    
    # --- Print the filtered Top 20 ---
    print(f"## Top {min(total_requested, MAX_RESULTS)} Largest DISCRETE Structural Variants (Filtered)")
    print(f"(Only includes EXACT types: {', '.join(DISCRETE_SV_TYPES)}; size $\ge$ 10 kb)\n")

    # Define the header format
    # Note: Since the only types included are exact matches, the CONSOLIDATED column will 
    # simply mirror the RAW_TYPE column, but we keep it for format consistency.
    header_top_n = f"{'Rank':<5}{'CHR':<12}{'START':>12}{'END':>12}{'RAW_TYPE':>8}{'SIZE_ABS':>12}{'CONSOLIDATED':>15}"
    separator_top_n = "-" * len(header_top_n)

    print(header_top_n)
    print(separator_top_n)

    # Print data rows
    if not top_n_rearrangements:
        print(f"No discrete structural variants of the requested EXACT types ($\ge$ {THRESHOLD_BP/1000:,} kb) were found.")
    else:
        for i, (chr_num, start, end, raw_sv_type, size) in enumerate(top_n_rearrangements):
            rank = i + 1
            # Since we are only keeping exact matches, CONSOLIDATED is the same as RAW_TYPE
            consolidated_type = raw_sv_type 
            # Use formatting for thousands separation
            print(f"{rank:<5}{chr_num:<12}{start:>12,}{end:>12,}{raw_sv_type:>8}{size:>12,}{consolidated_type:>15}")
    
    print("\n" + "=" * 80)


def main():
    """Main function to run the script."""
    # The script expects the file path as the first command-line argument
    if len(sys.argv) < 2:
        print("Usage: python sv_out_analyzer_strict.py <path_to_syri_out_file>")
        print("\nExample: python sv_out_analyzer_strict.py syri_final2.out")
        sys.exit(1)

    filepath = sys.argv[1]
    
    # Run processing and report generation
    rearrangements, exact_counts = read_and_process_sv_file(filepath)
    generate_report(rearrangements, exact_counts, filepath)

if __name__ == "__main__":
    main()

