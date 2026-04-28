import sys
import operator

# --- Configuration ---
SYRI_FILE = "syri.out" 
TARGET_SCAFFOLD = "scaffold_1" # Targeting the specific scaffold as requested
TOP_N = 3

# --- Helper Function ---
def calculate_length(start, end):
    """Calculates the length of the segment (end - start + 1)."""
    try:
        start = int(start)
        end = int(end)
        # Length calculation must handle inversions where end < start
        return abs(end - start) + 1
    except ValueError:
        return 0

# --- Main Analysis Function ---
def find_top_inversions(filename, scaffold_name, n):
    inversions = []
    found_count = 0
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split('\t')
                
                # Ensure the line has enough columns (at least 11 for SyRI output)
                if len(parts) < 11:
                    continue

                ref_chr = parts[0]
                sv_type = parts[10]

                # 1. Filter by Scaffold and Type (INV)
                if ref_chr == scaffold_name and sv_type == "INV":
                    found_count += 1
                    ref_start = parts[1]
                    ref_end = parts[2]
                    
                    length = calculate_length(ref_start, ref_end)
                    
                    # Store data: (length, line_index, full_line_parts)
                    inversions.append((length, len(inversions), parts))

        if not inversions:
            print(f"Error: No Inversions ('INV') found on {scaffold_name} in {filename}. Please check the scaffold name and ensure the file is present.")
            return

        # 2. Sort the inversions by length (descending)
        inversions.sort(key=operator.itemgetter(0), reverse=True)
        
        # 3. Get the top N
        top_inversions = inversions[:n]

        # 4. Print results
        print(f"\n--- Top {n} Largest Inversions on {scaffold_name} (Total INV found: {found_count}) ---")
        
        # Determine the length of the third largest inversion to estimate the visual cutoff
        visual_cutoff_estimate = top_inversions[-1][0] if len(top_inversions) == n else "N/A"

        for rank, (length, _, parts) in enumerate(top_inversions, 1):
            ref_start = parts[1]
            ref_end = parts[2]
            query_chr = parts[5]
            query_start = parts[6]
            query_end = parts[7]
            
            print(f"\n{rank}. Size: {length:,} bp")
            print(f"   Reference: {ref_chr}:{ref_start}-{ref_end}")
            print(f"   Query:     {query_chr}:{query_start}-{query_end}")
            print(f"   Annotation ID: {parts[9]}")
            print("-" * 20)
            
        print(f"\nBased on the plot showing only 3 inversions, the visual cutoff is likely around {visual_cutoff_estimate:,} bp.")


    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure '{filename}' is accessible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    find_top_inversions(SYRI_FILE, TARGET_SCAFFOLD, TOP_N)

