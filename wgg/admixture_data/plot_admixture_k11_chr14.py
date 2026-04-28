import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- USER-DEFINED FILE PATHS AND PARAMETERS ---

# Set the number of ancestral populations (K)
NUM_K = 11

# Path to the ADMIXTURE .q file for K=11 (UPDATED)
q_file_path = "admixture_results/admixture_chr14_inversion_final_ld_pruned.11.Q"

# Path to your population file
pop_file_path = "population_list.txt"

# Path to the .fam file to get individual IDs
fam_file_path = "admixture_chr14_inversion_final_ld_pruned.fam"

# Output file BASE NAME for the plot (UPDATED)
output_plot_base = "admixture_plot_K11_chr14"

# --- CUSTOM COLOR SCHEME (11 Tonalities of Cyan and Orange/Amber) ---
# Colors are applied sequentially to K1 through K11
colors = [
     # Orange Tonalities
    '#E65100',  # K4: Burnt Orange
    '#005C6B',  # K9: Very Dark Cyan (NEW) 
    '#FB8C00',  # K2: Standard Orange
    '#4DD0E1',  # K8: Light Cyan
    '#FFB300',  # K3: Golden Orange
    '#FF7043',  # K1: Vibrant Coral Orange
    '#80DEEA',  # K10: Pale Cyan (NEW)
    '#00ACC1',  # K7: Medium Cyan
    '#BF360C',  # K5: Deep Sienna
    '#FF9800',  # K6: Slightly Darker Orange (NEW)

    # Cyan Tonalities
#    '#00ACC1',  # K7: Medium Cyan
#    '#4DD0E1',  # K8: Light Cyan
#    '#005C6B',  # K9: Very Dark Cyan (NEW)
#    '#80DEEA',  # K10: Pale Cyan (NEW)
    
    # Amber/Yellow Tonalities (as a bridge between Orange and Cyan)
    '#FFD54F'   # K11: Light Amber/Yellow-Orange
]

# --- SCRIPT STARTS HERE ---

try:
    print(f"--- Preparing Data for ADMIXTURE Plot (K={NUM_K}) ---\n")

    # 1. Generate the list of K-component column names (DYNAMICALLY uses NUM_K)
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

    # 2. Load the K=11 admixture proportions and assign correct column names
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
    print(f"Original number of individuals: {initial_size}")
    print(f"Number of individuals to plot: {final_size}")
    if initial_size != final_size:
        print(f"✅ The script successfully filtered out {initial_size - final_size} individuals.\n")

    # Sort the individuals by their population for a clean plot
    merged_data.sort_values(by='Population', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # 3. Prepare data for plotting: Select ALL K=11 columns
    admixture_proportions = merged_data[K_COLUMNS].values

    # Get unique populations and their boundaries for tick marks
    unique_pops = merged_data['Population'].unique()
    pop_boundaries = [merged_data['Population'].eq(pop).sum() for pop in unique_pops]
    # Calculate the center point for the tick labels
    pop_boundaries_cum = np.cumsum(pop_boundaries) - (np.array(pop_boundaries) / 2)

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 6)) # Increased width for 11 components

    # 4. Create the stacked bar plot for all K components
    bottom = np.zeros(len(admixture_proportions))

    # Loop through all NUM_K components
    for i in range(NUM_K):
        # The color assignment is now fixed via the 'colors' list indices
        color_to_use = colors[i]
            
        ax.bar(
            range(len(admixture_proportions)), 
            admixture_proportions[:, i], 
            bottom=bottom, 
            color=color_to_use, 
            width=1.0, 
            # Note: We skip the label here as it clutters the plot. Population labels are used instead.
        )
        bottom += admixture_proportions[:, i]

    # Set x-axis ticks and labels
    ax.set_xticks(np.cumsum(pop_boundaries)[:-1]) 
    ax.set_xticklabels([]) # Hide individual tick labels

    # Add text labels at the center of each population group
    for pos, label in zip(pop_boundaries_cum, unique_pops):
        ax.text(pos, -0.05, label, ha='center', rotation=45, fontsize=10, transform=ax.get_xaxis_transform())


    # Add separating lines between populations
    for boundary in np.cumsum(pop_boundaries)[:-1]:
        ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5) 

    # Set y-axis labels and title
    ax.set_ylabel("Ancestry Proportion", fontsize=12)
    ax.set_title(f"ADMIXTURE Plot for K={NUM_K} (Chr14 Inversion Region)", fontsize=16)

    # Remove the default x-axis ticks/labels that were replaced by the text labels
    ax.tick_params(axis='x', which='major', length=0)
    ax.set_xlim(0, len(admixture_proportions)) # Ensure plot starts at 0 and ends at the last individual
    ax.set_ylim(0, 1)

    # Adjust plot layout and save
    plt.tight_layout()

    # --- SAVE BOTH PNG AND PDF FILES ---
    # Save as PNG with high DPI
    plt.savefig(f"{output_plot_base}.png", dpi=300)
    print(f"ADMIXTURE plot for K={NUM_K} saved to {output_plot_base}.png")

    # Save as PDF
    plt.savefig(f"{output_plot_base}.pdf")
    print(f"ADMIXTURE plot for K={NUM_K} saved to {output_plot_base}.pdf")

except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
