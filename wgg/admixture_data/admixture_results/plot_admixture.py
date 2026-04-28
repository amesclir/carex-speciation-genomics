plot_admixture.py 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os


# --- USER-DEFINED FILE PATHS AND PARAMETERS ---

# Path to the ADMIXTURE .q file for K=2

q_file_path = "admixture_results_CV/admixture_final_850k_pruned.2.Q"


# Path to your population file

pop_file_path = "population_list.txt"


# Path to the .fam file to get individual IDs

fam_file_path = "admixture_final_ld_pruned.fam"


# Output file name for the plot

output_plot_path = "admixture_plot_K2.png"


# --- SCRIPT STARTS HERE ---

try:

    print("--- Preparing Data for Plotting ---\n")


    # Load the population information file, which serves as our master list

    pop_data = pd.read_csv(pop_file_path, sep='\s+', header=0)

    pop_data.columns = ['Individual_ID', 'Population']


    # Load the .fam file and clean up the Individual IDs

    fam_data = pd.read_csv(fam_file_path, sep='\s+', header=None, usecols=[1])

    fam_data.columns = ['Individual_ID']

    fam_data['Individual_ID'] = fam_data['Individual_ID'].apply(

        lambda x: os.path.basename(x).rstrip('.bam')

    )


    # Load the K=2 admixture proportions

    q_data = pd.read_csv(q_file_path, sep='\s+', header=None)

    q_data.columns = [f'K2_Pop_{i+1}' for i in range(q_data.shape[1])]

    

    # Merge the fam data with q data and population data

    # The merge will automatically keep only the individuals present in all files

    merged_data = pd.concat([fam_data, q_data], axis=1)

    merged_data = pd.merge(merged_data, pop_data, on='Individual_ID', how='inner')

    

    # Check if the filtering worked correctly

    initial_size = len(fam_data)

    final_size = len(merged_data)

    print(f"Original number of individuals: {initial_size}")

    print(f"Number of individuals to plot: {final_size}")

    if initial_size != final_size:

        print(f"✅ The script successfully filtered out {initial_size - final_size} individuals.\n")


    # Sort the individuals by their population for a clean plot

    merged_data.sort_values(by='Population', inplace=True)

    

    # Prepare data for plotting

    admixture_proportions = merged_data[['K2_Pop_1', 'K2_Pop_2']].values

    

    # Get unique populations and their boundaries for tick marks

    unique_pops = merged_data['Population'].unique()

    pop_boundaries = [merged_data['Population'].eq(pop).sum() for pop in unique_pops]

    pop_boundaries_cum = np.cumsum(pop_boundaries) - (np.array(pop_boundaries) / 2)

    

    # Plotting

    fig, ax = plt.subplots(figsize=(15, 6))

    

    # Create the stacked bar plot

    bottom = np.zeros(len(admixture_proportions))

    colors = ['cyan', 'orange']

    

    for i in range(admixture_proportions.shape[1]):

        ax.bar(range(len(admixture_proportions)), admixture_proportions[:, i], bottom=bottom, color=colors[i], width=1.0)

        bottom += admixture_proportions[:, i]

    

    # Set x-axis ticks and labels

    ax.set_xticks(pop_boundaries_cum)

    ax.set_xticklabels(unique_pops, rotation=45, ha='right')

    ax.tick_params(axis='x', which='major', labelsize=10)

    

    # Add separating lines between populations

    for boundary in np.cumsum(pop_boundaries)[:-1]:

        ax.axvline(x=boundary, color='black', linestyle='--', linewidth=1)

    

    # Set y-axis labels and title

    ax.set_ylabel("Ancestry Proportion", fontsize=12)

    ax.set_title("ADMIXTURE Plot for K=2", fontsize=16)

    

    # Adjust plot layout and save

    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300)

    print(f"ADMIXTURE plot for K=2 saved to {output_plot_path}")


except FileNotFoundError as e:

    print(f"Error: {e}. Please check your file paths.")

except Exception as e:

    print(f"An unexpected error occurred: {e}") 
