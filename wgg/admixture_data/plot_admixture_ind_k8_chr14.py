import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- USER-DEFINED FILE PATHS AND PARAMETERS ---

# Set the number of ancestral populations (K)
NUM_K = 8

# Path to the ADMIXTURE .q file for K=8
q_file_path = "admixture_results/admixture_chr14_inversion_final_ld_pruned.8.Q"

# Path to your population file
pop_file_path = "population_list.txt"

# Path to the .fam file to get individual IDs
fam_file_path = "admixture_chr14_inversion_final_ld_pruned.fam"

# Output file BASE NAME for the plot (will be used for .png and .pdf)
output_plot_base = "admixture_plot_K8_chr14_individuals"

# Define 8 distinct colors based on the final requested scheme:
# K2, K6 (indices 1, 5) are Cyan shades.
# K1, K3, K4, K5, K7, K8 (indices 0, 2, 3, 4, 6, 7) are Orange/Amber shades.
colors = [
    # --- Cyan Tonalities ---
    '#FF7043',  # K1: Vibrant Coral Orange 
    '#00ACC1',  # K2: Medium Cyan
    
    # --- Orange Tonalities ---
    '#FB8C00',  # K3: Standard Orange (Swapped from K6)
    '#FFD54F',  # K4: Light Amber/Yellow-Orange
    '#FFB300',  # K5: Golden Orange
    
    # --- Cyan Tonalities ---
    '#4DD0E1',  # K6: Light Cyan (Swapped from K3)
    
    # --- Orange Tonalities ---
    '#E65100',  # K7: Burnt Orange
    '#BF360C'   # K8: Deep Sienna
]


# --- SCRIPT STARTS HERE ---
try:
    print(f"--- Preparing Data for ADMIXTURE Plot (K={NUM_K}) ---\n")

    # 1. Generate the list of K-component column names
    K_COLUMNS = [f'K{NUM_K}_Pop_{i+1}' for i in range(NUM_K)]

    # Load the population information file, which serves as our master list
    pop_data = pd.read_csv(pop_file_path, sep='\s+', header=0)
    pop_data.columns = ['Individual_ID', 'Population']

    # Load the .fam file and clean up the Individual IDs
    fam_data = pd.read_csv(fam_file_path, sep='\s+', header=None, usecols=[1])
    fam_data.columns = ['Individual_ID']
    fam_data['Individual_ID'] = fam_data['Individual_ID'].apply(
        lambda x: os.path.basename(x).rstrip('.bam')
    )

    # 2. Load the K=8 admixture proportions
    q_data = pd.read_csv(q_file_path, sep='\s+', header=None)
    
    # Ensure q_data has the correct number of columns before assignment
    if q_data.shape[1] != NUM_K:
        raise ValueError(f"Q-file has {q_data.shape[1]} columns, but script expected {NUM_K}.")
        
    q_data.columns = K_COLUMNS
    
    # Merge the fam data with q data and population data
    merged_data = pd.concat([fam_data, q_data], axis=1)
    merged_data = pd.merge(merged_data, pop_data, on='Individual_ID', how='inner')
    
    # Check data size
    initial_size = len(fam_data)
    final_size = len(merged_data)
    print(f"Number of individuals to plot: {final_size}\n")
    
    # Sort the individuals by their population for a clean plot
    merged_data.sort_values(by='Population', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
    
    # Prepare data for plotting
    admixture_proportions = merged_data[K_COLUMNS].values
    
    # Get unique populations and their boundaries for separating lines
    unique_pops = merged_data['Population'].unique()
    pop_boundaries = [merged_data['Population'].eq(pop).sum() for pop in unique_pops]
    pop_boundaries_cum = np.cumsum(pop_boundaries)[:-1]

    # Plotting
    # Use a dynamic figure size based on the number of individuals for better readability
    fig_height = max(10, final_size * 0.25)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    
    # Create the stacked horizontal bar plot
    left = np.zeros(len(admixture_proportions))
    
    for i in range(NUM_K):
        color_to_use = colors[i]
        
        # Plot K components as horizontal bars
        ax.barh(
            merged_data.index, 
            admixture_proportions[:, i], 
            left=left, 
            color=color_to_use, 
            height=1.0
        )
        left += admixture_proportions[:, i]
    
    # Set y-axis ticks and labels for individuals
    ax.set_yticks(range(len(merged_data)))
    ax.set_yticklabels(merged_data['Individual_ID'], fontsize=8)
    
    # Add separating lines between populations
    # We subtract 0.5 because the bar centers are at integer indices (0, 1, 2, ...)
    for boundary in pop_boundaries_cum:
        ax.axhline(y=boundary - 0.5, color='black', linestyle='--', linewidth=1)
    
    # Place population labels on the right-hand side of the plot
    # Calculate the center position for each population label
    pop_start_indices = [0] + np.cumsum(pop_boundaries).tolist()[:-1]
    
    for start_idx, size, pop_label in zip(pop_start_indices, pop_boundaries, unique_pops):
        center_position = start_idx + (size / 2)
        ax.text(
            1.01, 
            center_position, 
            pop_label, 
            verticalalignment='center', 
            rotation=270, 
            fontsize=12,
            transform=ax.get_yaxis_transform(), 
            color='black'
        )

    # Set x-axis labels and title
    ax.set_xlabel("Ancestry Proportion", fontsize=12)
    ax.set_ylabel("Individual ID", fontsize=12)
    ax.set_title(f"ADMIXTURE Plot for K={NUM_K} (Individual Level)", fontsize=16)
    ax.set_xlim(0, 1) # Proportions range from 0 to 1

    # Invert Y-axis so the first individual is at the top
    ax.invert_yaxis()

    # Adjust plot layout and save
    plt.tight_layout()
    
    # --- SAVE BOTH PNG AND PDF FILES ---
    # Save as PNG with high DPI
    plt.savefig(f"{output_plot_base}.png", dpi=300)
    print(f"ADMIXTURE plot with individual labels for K={NUM_K} saved to {output_plot_base}.png")
    
    # Save as PDF (vector format)
    plt.savefig(f"{output_plot_base}.pdf")
    print(f"ADMIXTURE plot with individual labels for K={NUM_K} saved to {output_plot_base}.pdf")

except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
