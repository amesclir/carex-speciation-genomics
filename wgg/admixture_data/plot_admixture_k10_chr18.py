import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- USER-DEFINED FILE PATHS AND PARAMETERS ---

# Set the number of ancestral populations (K)
NUM_K = 10

# Path to the ADMIXTURE .q file for K=10
q_file_path = "admixture_results/admixture_chr18_inversion_final_ld_pruned.10.Q"

# Path to your population file (Assumed to be the same)
pop_file_path = "population_list.txt"

# Path to the .fam file to get individual IDs
fam_file_path = "admixture_chr18_inversion_final_ld_pruned.fam"

# Output file BASE NAME for the plot
output_plot_base = "admixture_plot_K10_chr18_final_aligned"

# --- CUSTOM POPULATION ORDERING (REQUIRED) ---
# NOTE: The population listed FIRST will appear on the left (POP1).
CUSTOM_POP_ORDER = ['POP1_NAME_HERE', 'POP2_NAME_HERE'] 


# --- COMPONENT REORDERING MAP (CRITICAL FOR COLOR ALIGNMENT) ---
# >>> ACTION REQUIRED: UPDATE THIS ARRAY.
# The FIRST 3 indices (0, 1, 2) must be the original Q-file columns that are the true POP1 components (to get Cyan).
# The LAST 7 indices (3-9) must be the original Q-file columns that are the true POP2 components (to get Orange).
#
# Remember: Q-file columns are indexed 0 to 9.
K_REORDERING_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # <--- REPLACE THIS LIST

# --- CUSTOM COLOR SCHEME (Applied after reordering) ---
colors = [
    # Group 1 (K1-K3): Cyan Tonalities (Will plot the POP1 components from the map above)
'#BF360C',  # K8: Deep Sienna
'#E65100',  # K4: Burnt Orange (Deep)
#    '#005C6B',  # K1: Very Dark Cyan
#    '#4DD0E1',  # K2: Light Cyan
    '#00ACC1',  # K3: Medium Cyan
    
    # Group 2 (K4-K10): Orange/Amber Tonalities (Will plot the POP2 components from the map above)
'#005C6B',  # K1: Very Dark Cyan
#    '#E65100',  # K4: Burnt Orange (Deep)
    '#FB8C00',  # K5: Standard Orange
    '#FFB300',  # K6: Golden Orange
    '#FF7043',  # K7: Vibrant Coral Orange
'#4DD0E1',  # K2: Light Cyan
#    '#BF360C',  # K8: Deep Sienna
    '#FFD54F',  # K9: Amber/Yellow-Orange
    '#FF5722',  # K10: Deep Vivid Orange/Red-Orange
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

    # 2. Load the K=10 admixture proportions and assign correct column names
    q_data = pd.read_csv(q_file_path, sep='\s+', header=None)
    
    if q_data.shape[1] != NUM_K:
        raise ValueError(f"Q-file has {q_data.shape[1]} columns, but script expected {NUM_K}.")
        
    q_data.columns = K_COLUMNS

    # --- APPLY COMPONENT REORDERING ---
    if len(K_REORDERING_MAP) == NUM_K and all(i in K_REORDERING_MAP for i in range(NUM_K)):
        q_data_reordered = q_data.iloc[:, K_REORDERING_MAP]
        print("✅ Q-file columns reordered successfully based on K_REORDERING_MAP.")
    else:
        q_data_reordered = q_data.copy()
        print("⚠️ K_REORDERING_MAP is invalid or default, proceeding without reordering columns.")

    # Merge the fam data with q data and population data
    merged_data = pd.concat([fam_data, q_data_reordered], axis=1)
    merged_data = pd.merge(merged_data, pop_data, on='Individual_ID', how='inner')

    # --- CUSTOM POPULATION SORTING LOGIC ---
    unique_pops_found = merged_data['Population'].unique().tolist()
    
    final_pop_order = CUSTOM_POP_ORDER
    for p in unique_pops_found:
        if p not in final_pop_order:
            final_pop_order.append(p)

    pop_category = pd.CategoricalDtype(final_pop_order, ordered=True)
    merged_data['Population'] = merged_data['Population'].astype(pop_category)
    merged_data.sort_values(by='Population', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
    # --- END CUSTOM POPULATION SORTING LOGIC ---


    # 3. Prepare data for plotting: Select ALL K=10 columns (now reordered)
    admixture_proportions = merged_data[q_data_reordered.columns].values

    # Get unique populations and their boundaries for tick marks
    unique_pops = merged_data['Population'].unique()
    pop_boundaries = [merged_data['Population'].eq(pop).sum() for pop in unique_pops]
    pop_boundaries_cum = np.cumsum(pop_boundaries) - (np.array(pop_boundaries) / 2)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 6))

    # 4. Create the stacked bar plot for all K components
    bottom = np.zeros(len(admixture_proportions))

    # Loop through all NUM_K components
    # Colors are applied based on the index (0-9) of the *reordered* column
    for i in range(NUM_K):
        color_to_use = colors[i]
            
        ax.bar(
            range(len(admixture_proportions)), 
            admixture_proportions[:, i], 
            bottom=bottom, 
            color=color_to_use, 
            width=1.0, 
        )
        bottom += admixture_proportions[:, i]

    # Set x-axis ticks and labels
    ax.set_xticks(np.cumsum(pop_boundaries)[:-1]) 
    ax.set_xticklabels([]) 

    # Add text labels at the center of each population group
    for pos, label in zip(pop_boundaries_cum, unique_pops):
        ax.text(pos, -0.05, label, ha='center', rotation=45, fontsize=10, transform=ax.get_xaxis_transform())

    # Add separating lines between populations
    for boundary in np.cumsum(pop_boundaries)[:-1]:
        ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5) 

    # Set y-axis labels and title
    ax.set_ylabel("Ancestry Proportion", fontsize=12)
    ax.set_title(f"ADMIXTURE Plot for K={NUM_K} (Chr18 Inversion Region - Final Aligned)", fontsize=16)

    ax.tick_params(axis='x', which='major', length=0)
    ax.set_xlim(0, len(admixture_proportions)) 
    ax.set_ylim(0, 1)

    # Adjust plot layout and save
    plt.tight_layout()

    # --- SAVE BOTH PNG AND PDF FILES ---
    plt.savefig(f"{output_plot_base}.png", dpi=300)
    print(f"ADMIXTURE plot for K={NUM_K} saved to {output_plot_base}.png")

    plt.savefig(f"{output_plot_base}.pdf")
    print(f"ADMIXTURE plot for K={NUM_K} saved to {output_plot_base}.pdf")

except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
