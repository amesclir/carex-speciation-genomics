import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- USER-DEFINED FILE PATHS ---
fst_file_path = "admixture_final_ld_pruned_filtered_pop1_vs_pop2.fst"
syri_file_path = "/home/aescudero/syri/syri.out"

# --- SECTION: Load and process Fst data ---
df_fst = pd.read_csv(fst_file_path, sep='\s+')
df_fst.dropna(subset=['FST'], inplace=True)
df_fst['FST'] = df_fst['FST'].clip(lower=0)
df_fst = df_fst[df_fst['FST'] > 0.9]
df_fst['CHR_int'] = df_fst['CHR'].astype(int)
df_fst['color'] = np.where(df_fst['CHR_int'] % 2 == 1, 'gray', 'black')

# Create a 'BPcum' column for the continuous x-axis
df_fst['base_pos'] = 0
last_pos = 0
for chr_num in df_fst['CHR_int'].unique():
    chr_indices = df_fst['CHR_int'] == chr_num
    df_fst.loc[chr_indices, 'base_pos'] = last_pos
    last_pos += df_fst.loc[chr_indices, 'POS'].max()
df_fst['BPcum'] = df_fst['base_pos'] + df_fst['POS']

# --- SECTION: Load and process SyRI data with new parsing logic ---
df_raw = pd.read_csv(syri_file_path, sep='\s+', header=None)

# We'll use the 11th column for the TYPE, as it appears to be consistent.
df_raw = df_raw.rename(columns={10: 'TYPE'})

print("\n--- Summary of all SyRI rearrangement types ---")
print(df_raw['TYPE'].value_counts())
print("--------------------------------------------------\n")

# --- MODIFIED: Filter for specific rearrangement types ---
# Define the new, restricted list of rearrangement types
rearrangement_types = ['INV', 'INVTR', 'INVDPAL', 'INVTRAL', 'DUP', 'DUPAL', 'TRANS', 'TRANSAL']

df_rearrangements = df_raw[df_raw['TYPE'].isin(rearrangement_types)].copy()

# Combine data from two possible formats
df_rearrangements['CHR'] = np.where(df_rearrangements[0] != '-', df_rearrangements[0], df_rearrangements[5])
df_rearrangements['START'] = np.where(df_rearrangements[1] != '-', df_rearrangements[1], df_rearrangements[6])
df_rearrangements['END'] = np.where(df_rearrangements[2] != '-', df_rearrangements[2], df_rearrangements[7])

# Filter out rows where CHR, START, or END are still invalid
df_rearrangements = df_rearrangements[df_rearrangements['CHR'].str.contains('scaffold_')].copy()

# Convert positions to numeric, coercing errors to NaN
df_rearrangements['START'] = pd.to_numeric(df_rearrangements['START'], errors='coerce')
df_rearrangements['END'] = pd.to_numeric(df_rearrangements['END'], errors='coerce')
df_rearrangements.dropna(subset=['START', 'END'], inplace=True)

# Calculate rearrangement size and filter
df_rearrangements['SIZE'] = df_rearrangements['END'] - df_rearrangements['START']
SIZE_THRESHOLD = 10000
df_rearrangements = df_rearrangements[df_rearrangements['SIZE'].abs() > SIZE_THRESHOLD].copy()

print(f"Plotting {len(df_rearrangements)} rearrangements with size > {SIZE_THRESHOLD} bp.")

print("\n--- First 5 rearrangements to be plotted ---")
print(df_rearrangements[['CHR', 'START', 'END', 'TYPE', 'SIZE']].head().to_string())
print("------------------------------------------\n")

# Add a cumulative position to the syri data for plotting
df_rearrangements['CHR_int'] = df_rearrangements['CHR'].str.replace('scaffold_', '').astype(int)
df_rearrangements = pd.merge(df_rearrangements, df_fst[['CHR_int', 'base_pos']].drop_duplicates(), on='CHR_int', how='left')
df_rearrangements.dropna(subset=['base_pos'], inplace=True)

df_rearrangements['START_cum'] = df_rearrangements['base_pos'] + df_rearrangements['START']
df_rearrangements['END_cum'] = df_rearrangements['base_pos'] + df_rearrangements['END']

# --- SECTION: Plotting the data ---
fig, ax = plt.subplots(figsize=(15, 8))

has_rearrangements_to_plot = not df_rearrangements.empty
if has_rearrangements_to_plot:
    for index, row in df_rearrangements.iterrows():
        ax.axvspan(row['START_cum'], row['END_cum'], color='pink', alpha=0.3, zorder=0)
    ax.axvspan(0, 0, color='pink', alpha=0.3, label=f'Rearrangements (>{SIZE_THRESHOLD} bp)')

ax.scatter(x=df_fst['BPcum'], y=df_fst['FST'], s=0.5, c=df_fst['color'], zorder=1, label='Fst Loci > 0.9')

threshold = df_fst['FST'].quantile(0.99)
df_outliers = df_fst[df_fst['FST'] > threshold]
ax.scatter(x=df_outliers['BPcum'], y=df_outliers['FST'], s=10, c='red', zorder=2, label=f'Top 1% Fst Loci (>{threshold:.2f})')

chrom_df = df_fst.groupby('CHR_int')['BPcum'].agg(['min', 'max'])
chrom_df['mid'] = (chrom_df['min'] + chrom_df['max']) / 2
ax.set_xticks(chrom_df['mid'])
ax.set_xticklabels(chrom_df.index)

ax.set_ylim(0.9, 1.0)
ax.set_xlabel("Chromosome", fontsize=12)
ax.set_ylabel("Fst", fontsize=12)
ax.set_title(f"Fst Values for POP1 vs POP2 (Fst > 0.9)", fontsize=16)

ax.legend()
plt.savefig(f"fst_manhattan_plot_high_Fst_10k_rearrangements.png")
