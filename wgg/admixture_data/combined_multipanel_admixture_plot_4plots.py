import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- USER-DEFINED GLOBAL FILE PATHS AND PARAMETERS ---
pop_file_path = "population_list.txt" # Path to your population file (common for all)
output_plot_base = "admixture_k8_k11_k14_subset_plot" # NEW BASE FILENAME for the 4 requested plots

# --- CUSTOM POPULATION ORDERING (REQUIRED) ---
# NOTE: Individuals will be sorted within these groups.
CUSTOM_POP_ORDER = ['Carex borbonica', 'Carex boryana'] 
# Other populations found in population_list.txt will be appended after these.

# --- DATASET CONFIGURATION LIST (Subsetted to K=8, K=11, K=14 (Chr20), K=14 (Chr32)) ---
PLOTS_CONFIG = [
    # 1. K=8 (Whole Dataset)
    {
        'title': "ADMIXTURE Plot K=8 (Whole Dataset)",
        'k': 8,
        'q_path': "admixture_results_cv/admixture_final_850k_pruned.8.Q",
        'fam_path': "admixture_final_850k_pruned.fam",
        'reordering_map': list(range(8)), 
        'colors': [
            '#00838F', # K1: Dark Cyan/Teal (Cyan)
            '#00ACC1', # K2: Medium Cyan (Cyan)
            '#FB8C00', # K3: Standard Orange
            '#FFD54F', # K4: Light Amber/Yellow-Orange
            '#FFB300', # K5: Golden Orange
            '#4DD0E1', # K6: Light Cyan (Cyan)
            '#E65100', # K7: Burnt Orange
            '#BF360C', # K8: Deep Sienna
        ]
    },
    # 2. K=11 (Chr14 Inversion)
    {
        'title': "ADMIXTURE Plot K=11 (Chr14 Inversion Region)",
        'k': 11,
        'q_path': "admixture_results/admixture_chr14_inversion_final_ld_pruned.11.Q",
        'fam_path': "admixture_chr14_inversion_final_ld_pruned.fam",
        'reordering_map': list(range(11)), 
        'colors': [
            '#E65100', # K1: Orange
            '#005C6B', # K2: Cyan
            '#FB8C00', # K3: Orange
            '#4DD0E1', # K4: Cyan
            '#FFB300', # K5: Orange
            '#FF7043', # K6: Orange
            '#80DEEA', # K7: Cyan
            '#00ACC1', # K8: Cyan
            '#BF360C', # K9: Orange
            '#FF9800', # K10: Orange
            '#FFD54F'  # K11: Amber/Yellow-Orange
        ]
    },
    # 3. K=14 (Chr20 Inversion)
    {
        'title': "ADMIXTURE Plot K=14 (Chr20 Inversion Region)",
        'k': 14,
        'q_path': "admixture_results/admixture_chr20_inversion_final_ld_pruned.14.Q",
        'fam_path': "admixture_chr20_inversion_final_ld_pruned.fam",
        'reordering_map': list(range(14)), 
        'colors': [
            '#BF360C', # K1: Orange
            '#E65100', # K2: Orange
            '#005C6B', # K3: Cyan
            '#FB8C00', # K4: Orange
            '#FFB300', # K5: Orange
            '#4DD0E1', # K6: Cyan
            '#00ACC1', # K7: Cyan
            '#FF7043', # K8: Orange
            '#78C2AD', # K9: Cyan
            '#FFD54F', # K10: Orange
            '#A2D9CE', # K11: Cyan
            '#FF5722', # K12: Orange
            '#D68910', # K13: Orange
            '#D45A25', # K14: Orange
        ]
    },
    # 4. K=14 (Chr32 Inversion)
    {
        'title': "ADMIXTURE Plot K=14 (Chr32 Inversion Region)",
        'k': 14,
        'q_path': "admixture_results/admixture_chr32_inversion_final_ld_pruned.14.Q",
        'fam_path': "admixture_chr32_inversion_final_ld_pruned.fam",
        'reordering_map': list(range(14)), 
        'colors': [
            '#BF360C', # K1: Orange
            '#E65100', # K2: Orange
            '#005C6B', # K3: CYAN
            '#FB8C00', # K4: Orange
            '#FFB300', # K5: Orange
            '#FF7043', # K6: Orange
            '#78C2AD', # K7: CYAN
            '#FFD54F', # K8: Orange
            '#A2D9CE', # K9: CYAN
            '#3E8E7E', # K10: CYAN
            '#FF5722', # K11: Orange
            '#D68910', # K12: Orange
            '#D45A25', # K13: Orange
            '#C63748', # K14: Orange
        ]
    },
]

# --- FUNCTIONS ---

def load_and_prepare_data(config):
    """
    Loads FAM, Q, and Population data, merges them, applies reordering, 
    and sorts based on the population list defined at the top of the script.
    """
    num_k = config['k']
    q_file_path = config['q_path']
    fam_file_path = config['fam_path']
    reordering_map = config['reordering_map']

    print(f"Loading data for K={num_k} from {os.path.basename(q_file_path)}...")
    
    # 0. ADDED CHECK: Validate K against the number of defined colors
    if len(config['colors']) != num_k:
        print(f"Error: Color list length ({len(config['colors'])}) mismatch K ({num_k}) for {config['title']}. Skipping this plot.")
        return None, None, None, None


    # 1. Load Population list (master list)
    try:
        pop_data = pd.read_csv(pop_file_path, sep='\s+', header=0)
        pop_data.columns = ['Individual_ID', 'Population']
    except FileNotFoundError:
        print(f"Error: Population file not found at {pop_file_path}. Skipping this plot.")
        return None, None, None, None

    # 2. Load .fam file and clean up Individual IDs
    try:
        fam_data = pd.read_csv(fam_file_path, sep='\s+', header=None, usecols=[1])
    except FileNotFoundError:
        print(f"Error: FAM file not found at {fam_file_path}. Skipping this plot.")
        return None, None, None, None
    
    fam_data.columns = ['Individual_ID']
    fam_data['Individual_ID'] = fam_data['Individual_ID'].apply(
        lambda x: os.path.basename(x).rstrip('.bam')
    )

    # 3. Load Q-file and check K-value consistency
    try:
        q_data = pd.read_csv(q_file_path, sep='\s+', header=None)
    except FileNotFoundError:
        print(f"Error: Q-file not found at {q_file_path}. Skipping this plot.")
        return None, None, None, None

    if q_data.shape[1] != num_k:
        print(f"Error: Q-file columns ({q_data.shape[1]}) mismatch K ({num_k}). Skipping this plot.")
        return None, None, None, None
        
    K_COLUMNS = [f'K{num_k}_Pop_{i+1}' for i in range(num_k)]
    q_data.columns = K_COLUMNS
    
    # 4. Apply Component Reordering (CRITICAL STEP)
    if len(reordering_map) == num_k and all(i in reordering_map for i in range(num_k)):
        q_data_reordered = q_data.iloc[:, reordering_map]
    else:
        q_data_reordered = q_data.copy()
        print(f"Warning: K={num_k} reordering map is invalid or default. Using original column order.")

    # 5. Merge Data
    merged_data = pd.concat([fam_data, q_data_reordered], axis=1)
    merged_data = pd.merge(merged_data, pop_data, on='Individual_ID', how='inner')

    # 6. Apply Population Sorting Logic (Ensures consistent order for visible populations)
    unique_pops_found = merged_data['Population'].unique().tolist()
    
    final_pop_order = [p for p in CUSTOM_POP_ORDER if p in unique_pops_found]
    for p in unique_pops_found:
        if p not in final_pop_order:
            final_pop_order.append(p) # Append any unlisted populations

    pop_category = pd.CategoricalDtype(final_pop_order, ordered=True)
    merged_data['Population'] = merged_data['Population'].astype(pop_category)
    merged_data.sort_values(by='Population', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # 7. Prepare output for plotting
    admixture_proportions = merged_data[q_data_reordered.columns].values
    
    # Calculate unique populations and boundaries for tick marks
    unique_pops = merged_data['Population'].unique()
    pop_boundaries = [merged_data['Population'].eq(pop).sum() for pop in unique_pops]
    pop_boundaries_cum = np.cumsum(pop_boundaries) - (np.array(pop_boundaries) / 2)

    return admixture_proportions, pop_boundaries, pop_boundaries_cum, unique_pops

def generate_multi_panel_plot():
    """
    Creates a single figure with multiple subplots based on the PLOTS_CONFIG.
    """
    num_plots = len(PLOTS_CONFIG)
    
    # Create the figure and subplots. Increased height (4) and wider figure (20) 
    fig, axes = plt.subplots(
        nrows=num_plots, 
        ncols=1, 
        figsize=(20, 4 * num_plots), # 4 inches height per plot
        sharex=False, 
        gridspec_kw={'hspace': 0.15} # Adjust vertical spacing between subplots
    )
    
    if num_plots == 1:
        axes = [axes] # Ensure axes is iterable even for a single plot

    all_data = []
    
    # --- 1. Load Data for All Plots ---
    for config in PLOTS_CONFIG:
        data = load_and_prepare_data(config)
        all_data.append(data)

    # Find the maximum number of individuals across all plots to set a consistent X-limit
    valid_data = [data[0] for data in all_data if data[0] is not None]
    max_individuals = max([data.shape[0] for data in valid_data] or [0])
    
    if max_individuals == 0:
        print("No valid data loaded for plotting. Exiting.")
        return

    # --- 2. Plotting Loop ---
    for idx, (ax, config) in enumerate(zip(axes, PLOTS_CONFIG)):
        proportions, pop_boundaries, pop_boundaries_cum, unique_pops = all_data[idx]

        if proportions is None:
            # If data loading failed, display a simple message instead of crashing.
            ax.set_title(f"DATA ERROR: {config['title']}", loc='left', color='red')
            ax.axis('off')
            continue

        num_k = config['k']
        colors = config['colors']

        bottom = np.zeros(len(proportions))
        
        # ALIGNMENT CORRECTION: Shift x-indices by +0.5 
        x_indices = np.arange(len(proportions)) + 0.5

        # Create the stacked bar plot
        for i in range(num_k):
            color_to_use = colors[i]
            ax.bar(
                x_indices, # Use the shifted indices for perfect alignment
                proportions[:, i],
                bottom=bottom,
                color=color_to_use,
                width=1.0,
            )
            bottom += proportions[:, i]

        # --- Plot Aesthetics ---
        
        # Add separating lines between populations.
        for boundary in np.cumsum(pop_boundaries)[:-1]:
            ax.axvline(x=boundary, color='black', linestyle='-', linewidth=1.5)

        # Title and Labels
        ax.set_title(config['title'], fontsize=14, loc='left')
        ax.set_ylabel(f"Ancestry Prop. (K={num_k})", fontsize=10)
        
        # X-axis cleanup: Hide ticks and labels
        ax.set_xticks(np.cumsum(pop_boundaries)[:-1]) 
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='major', length=0)
        # Set x-limits to encompass the shifted bars correctly
        ax.set_xlim(0, len(proportions)) 
        ax.set_ylim(0, 1)

        # Add text labels only to the bottom plot
        if idx == num_plots - 1:
            for pos, label in zip(pop_boundaries_cum, unique_pops):
                # Position adjusted to -0.12 to prevent overlap
                ax.text(pos, -0.12, label, ha='center', rotation=45, fontsize=10, transform=ax.get_xaxis_transform())
        else:
             # Hide x-axis ticks and labels for all other plots
             ax.set_xticks([])
             ax.set_xticklabels([])


        # --- Add Legend (Specific to each K) ---
        legend_handles = [plt.matplotlib.patches.Patch(color=colors[i]) for i in range(num_k)]
        legend_labels = [f'K{i+1}' for i in range(num_k)]

        # Place the legend outside the plot area on the right
        ax.legend(
            legend_handles,
            legend_labels,
            title=f"Components (K={num_k})",
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.,
            ncol=1,
            fontsize=9
        )
    
    # Adjust layout to make room for all the legends on the right side
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 

    # --- SAVE BOTH PNG AND PDF FILES ---
    plt.savefig(f"{output_plot_base}.png", dpi=300)
    print(f"\n✅ Combined ADMIXTURE plot saved to {output_plot_base}.png")

    plt.savefig(f"{output_plot_base}.pdf")
    print(f"✅ Combined ADMIXTURE plot saved to {output_plot_base}.pdf")
    

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    try:
        generate_multi_panel_plot()
    except Exception as e:
        # Catch any remaining unexpected errors
        print(f"An unexpected error occurred during plot generation: {e}")
