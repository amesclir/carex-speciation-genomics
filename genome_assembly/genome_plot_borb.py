import json
from collections import defaultdict
import math
import os
import random
import sys

# --- 1. Configuration & File Paths (UPDATED for C. borbonica) ---
FASTA_FAI_PATH = '/home/aescudero/genome_assembly/B6_clean_assemblies/C_borbonica/C_borbonica.fasta.fai'

# NOTE: Using the .gtf file for gene annotations as found in your directory
GENE_GFF_PATH = '/home/aescudero/genome_assembly/C2_braker/braker_cborb/braker.gtf' 
REPEAT_GFF_PATH = '/home/aescudero/genome_assembly/C3_earlgrey/C_borbonica/C_borbonica_EarlGrey/C_borbonica_summaryFiles/C_borbonica.filteredRepeats.gff'
OUTPUT_FILENAME = 'carex_borbonica_genomic_data.json' # Output filename changed

# Fixed window size for genomic bins (Updated to 100,000 bp / 0.1 Mb)
FIXED_BIN_SIZE_BP = 100000

# --- 2. Track Definitions ---
GENE_TRACK = 'Gene Density'

# Define the specific repeat element tracks requested (Copia, Gypsy, Satellite, DNA)
REPEAT_TRACKS = {
    'LTR/Copia': 'LTR/Copia',
    'LTR/Gypsy': 'LTR/Gypsy',
    'Satellite DNA': 'Satellite DNA',
    'DNA Transposons': 'DNA Transposons',
}

# --- 3. Load Contig Lengths ---
def load_contig_lengths(fai_path):
    """
    Loads contig lengths from a FASTA index (.fai) file.
    """
    contig_lengths = {}

    try:
        with open(fai_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        contig_id = parts[0]
                        length = int(parts[1])
                        contig_lengths[contig_id] = length
                    except ValueError:
                        print(f"Warning: Skipping line in FAI file due to non-integer length: {line.strip()}")
                        continue

        if not contig_lengths:
            print(f"FATAL ERROR: FAI file at {fai_path} found but contained no valid scaffold data.")
            sys.exit(1)
            
        print(f"SUCCESS: Loaded lengths for {len(contig_lengths)} scaffolds from {fai_path}.")
        return contig_lengths
        
    except FileNotFoundError:
        print(f"FATAL ERROR: FAI file not found at {fai_path}.")
        print("Please check and ensure the .fai file exists in that directory.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while reading the FAI file: {e}")
        sys.exit(1)


# --- 4. BIN DEFINITION ---

def create_genomic_bins(chr_lengths, fixed_bin_size_bp):
    """
    Creates a list of all genomic bins with their coordinates based on a fixed size.
    """
    bins = {}
    for chr_id, length in chr_lengths.items():
        if length == 0: continue
        num_bins = math.ceil(length / fixed_bin_size_bp)
        
        chr_bins = []
        for i in range(num_bins):
            start = i * fixed_bin_size_bp + 1
            # The last bin takes all remaining length
            end = min((i + 1) * fixed_bin_size_bp, length)
            
            chr_bins.append({'start': start, 'end': end, 'size': end - start + 1})
        bins[chr_id] = chr_bins
    return bins

# --- 5. FEATURE PROCESSING FUNCTIONS ---

def calculate_gene_density(gff_path, genomic_bins):
    """Calculates gene count per bin (normalized). Works with GFF/GTF features marked 'gene'."""
    bin_gene_counts = defaultdict(lambda: defaultdict(int))
    max_gene_count = 0
    
    try:
        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                try:
                    parts = line.split('\t')
                    # Check for minimal GFF/GTF fields and feature type
                    # The gene type is expected in the 3rd column (index 2)
                    if len(parts) < 5 or parts[2] != 'gene': continue 

                    chr_id, feature_type, start, end = parts[0], parts[2], int(parts[3]), int(parts[4])
                except (IndexError, ValueError):
                    continue

                if chr_id in genomic_bins:
                    # Calculate start bin index
                    start_bin_idx = math.floor((start - 1) / FIXED_BIN_SIZE_BP)
                    
                    # We only care about the starting bin for counting genes once
                    if start_bin_idx < len(genomic_bins[chr_id]):
                        bin_gene_counts[chr_id][start_bin_idx] += 1
                        max_gene_count = max(max_gene_count, bin_gene_counts[chr_id][start_bin_idx])

    except FileNotFoundError:
        print(f"ERROR: Gene file not found at {gff_path}")
        return []

    # Convert counts to normalized density
    density_output = []
    # Sort scaffolds numerically
    sorted_scaffolds = sorted(genomic_bins.keys(), key=lambda x: (x.startswith('scaffold_'), int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')))

    for chr_id in sorted_scaffolds:
        for i, bin_info in enumerate(genomic_bins[chr_id]):
            count = bin_gene_counts[chr_id][i]
            density = count / max_gene_count if max_gene_count > 0 else 0
            density_output.append({
                'chr': chr_id,
                'bin_index': i,
                'track': GENE_TRACK,
                'density': density
            })
    return density_output

def calculate_repeat_density(gff_path, genomic_bins):
    """Calculates repeat coverage fraction per bin for specific user-requested transposons."""
    # bin_coverage[chr_id][bin_index][track_name] = overlap_length
    bin_coverage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Get the defined track names
    track_names = list(REPEAT_TRACKS.keys())

    try:
        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                try:
                    parts = line.split('\t')
                    # Check for minimal GFF fields
                    if len(parts) < 5: continue
                        
                    chr_id, _, repeat_class, start, end = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
                except (IndexError, ValueError):
                    continue
                
                # --- LOGIC TO MAP REPEAT CLASS TO REQUESTED TRACKS ---
                track_name = None
                
                # Check for Copia, Gypsy, Satellite, or DNA
                if 'LTR/Copia' in repeat_class:
                    track_name = 'LTR/Copia'
                elif 'LTR/Gypsy' in repeat_class:
                    track_name = 'LTR/Gypsy'
                elif 'Satellite' in repeat_class:
                    track_name = 'Satellite DNA'
                elif repeat_class.startswith('DNA'): # Covers 'DNA', 'DNA/Subtype', etc.
                    track_name = 'DNA Transposons'

                if not track_name: continue
                # --------------------------------------------------------

                if chr_id in genomic_bins:
                    # Calculate start and end bin index
                    start_bin_idx = math.floor((start - 1) / FIXED_BIN_SIZE_BP)
                    end_bin_idx = math.floor((end - 1) / FIXED_BIN_SIZE_BP)
                    
                    # Iterate over all bins the repeat element covers
                    for i in range(start_bin_idx, end_bin_idx + 1):
                        if i >= len(genomic_bins[chr_id]): continue # Safety break if bin index is out of range
                        
                        bin_info = genomic_bins[chr_id][i]
                        
                        # Calculate overlap with the current bin
                        overlap_start = max(start, bin_info['start'])
                        overlap_end = min(end, bin_info['end'])
                        overlap_length = max(0, overlap_end - overlap_start + 1)
                        
                        if overlap_length > 0:
                            bin_coverage[chr_id][i][track_name] += overlap_length

    except FileNotFoundError:
        print(f"ERROR: Repeat file not found at {gff_path}")
        return []

    # Convert coverage length to density (fraction of bin size)
    density_output = []
    sorted_scaffolds = sorted(genomic_bins.keys(), key=lambda x: (x.startswith('scaffold_'), int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf')))

    for chr_id in sorted_scaffolds:
        for i, bin_info in enumerate(genomic_bins[chr_id]):
            bin_size = bin_info['size']
            for track_name in REPEAT_TRACKS.values():
                coverage_length = bin_coverage[chr_id][i][track_name]
                # Density is coverage length divided by the bin size
                density = coverage_length / bin_size
                density_output.append({
                    'chr': chr_id,
                    'bin_index': i,
                    'track': track_name,
                    'density': min(1.0, round(density, 4)) # Cap at 1.0 and round
                })

    return density_output

# --- 6. MAIN EXECUTION ---

def generate_density_plot_data():
    
    # 1. Load Contig Lengths
    chr_lengths = load_contig_lengths(FASTA_FAI_PATH)
    
    if chr_lengths is None:
        return # Exit gracefully due to FAI error

    total_genome_size = sum(chr_lengths.values())
    
    print(f"Starting data processing for {len(chr_lengths)} scaffolds...")
    print(f"Total genome size: {total_genome_size / 1000000:.1f} Mb")
    print(f"Using a fixed bin size of {FIXED_BIN_SIZE_BP:,} bp (0.1 Mb).")
    
    # 2. Define all 0.1Mb bins
    genomic_bins = create_genomic_bins(chr_lengths, FIXED_BIN_SIZE_BP)
    total_bins = sum(len(bins) for bins in genomic_bins.values())
    print(f"Defined {total_bins} total bins across the genome.")

    # 3. Calculate Gene Density
    gene_densities = calculate_gene_density(GENE_GFF_PATH, genomic_bins)
    
    # 4. Calculate Repeat Densities
    repeat_densities = calculate_repeat_density(REPEAT_GFF_PATH, genomic_bins)

    # 5. Combine and Save
    final_data = gene_densities + repeat_densities

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\nSuccessfully generated and saved data to {OUTPUT_FILENAME}")
    print(f"Total data points (bins * tracks): {len(final_data)}")

if __name__ == "__main__":
    generate_density_plot_data()
