# --- Circos Plot for Borbonica Genomic Density (5 Tracks) ---
# This script uses the 'circlize' R package to visualize data for 1 Gene track and 4 Repeat tracks.

# 1. Install and Load Required Packages
# Run these commands in your R console if the packages are not already installed:
# install.packages(c("circlize", "jsonlite", "dplyr"))

library(circlize)
library(jsonlite)
library(dplyr)
library(RColorBrewer)

# --- Configuration ---
DATA_FILE <- "carex_borbonica_genomic_data.json"
BIN_SIZE_BP <- 100000 # Must match the Python script

# --- Color Definitions for all 5 Tracks ---
GENE_COLOR      <- "#10b981" # Emerald (Gene Density)
COPIA_COLOR     <- "#f97316" # Orange (LTR/Copia)
GYPSY_COLOR     <- "#ef4444" # Red (LTR/Gypsy)
SATELLITE_COLOR <- "#8b5cf6" # Violet (Satellite DNA)
DNA_COLOR       <- "#3b82f6" # Blue (DNA Transposons)

# --- Data Loading and Preparation ---

# Load the JSON data
cat("Loading data from:", DATA_FILE, "\n")
data_raw <- fromJSON(DATA_FILE)

# 1. Calculate genomic coordinates (Start/End BP) for each bin
plot_data <- data_raw %>%
  mutate(
    # Calculate start_bp and end_bp first!
    start_bp = bin_index * BIN_SIZE_BP,
    end_bp = (bin_index + 1) * BIN_SIZE_BP,
    # Calculate mid_bp using the newly created start_bp
    mid_bp = start_bp + (BIN_SIZE_BP / 2) 
  )

# 2. Get the maximum length (in BP) for each scaffold
chr_len_df <- plot_data %>%
  group_by(chr) %>%
  summarise(
    max_bin_index = max(bin_index),
    length = (max_bin_index + 1) * BIN_SIZE_BP
  ) %>%
  ungroup()

# Sort scaffolds numerically
chr_len_df <- chr_len_df %>%
  arrange(as.numeric(sub("scaffold_", "", chr)))

# Prepare the scaffold length vector for circos.initialize
chr_lengths <- setNames(chr_len_df$length, chr_len_df$chr)


# 3. Split data into separate data frames for each track
genes_df      <- plot_data %>% filter(track == "Gene Density")
copia_df      <- plot_data %>% filter(track == "LTR/Copia")
gypsy_df      <- plot_data %>% filter(track == "LTR/Gypsy")
satellite_df  <- plot_data %>% filter(track == "Satellite DNA")
dna_df        <- plot_data %>% filter(track == "DNA Transposons")

# 4. Calculate overall mean density for baselines (for visual context)
mean_gene      <- mean(genes_df$density)
mean_copia     <- mean(copia_df$density)
mean_gypsy     <- mean(gypsy_df$density)
mean_satellite <- mean(satellite_df$density)
mean_dna       <- mean(dna_df$density)


# --- Circos Plotting ---

# 5. Initialize the Circos plot
circos.clear()
# Set a unique name for the new plot file
pdf("carex_borbonica_circos_plot_line.pdf", width = 9, height = 9)

# Set plot parameters: gap size, track margin
circos.par(
  gap.degree = 1.5,
  cell.padding = c(0.02, 0, 0.02, 0),
  track.margin = c(0.01, 0.01)
)

circos.initialize(
  factors = names(chr_lengths),
  xlim = cbind(rep(0, length(chr_lengths)), chr_lengths)
)

# 6. Karyotype (Scaffold Names and Background) - Outer Track
circos.track(
  ylim = c(0, 1),
  panel.fun = function(x, y) {
    circos.rect(
      xleft = CELL_META$xlim[1], xright = CELL_META$xlim[2],
      ybottom = 0, ytop = 1,
      col = "grey90", border = NA
    )

    name = CELL_META$sector.index
    name_abbr = sub("scaffold_", "S", name)
    
    circos.text(
      x = CELL_META$xcenter, y = CELL_META$ylim[2] + mm_y(2),
      labels = name_abbr,
      facing = "clockwise", niceFacing = TRUE, cex = 0.7,
      adj = c(0, 0.5)
    )
  },
  track.height = 0.05,
  bg.border = NA
)

# --- Plotting Data Tracks (Inner 5 Tracks) ---

# Track 1 (Innermost): Gene Density (Using Lines)
circos.track(
  ylim = c(0, 1),
  track.height = 0.08, 
  bg.col = "white",
  bg.border = "grey50",
  panel.fun = function(x, y) {
    chr_name = CELL_META$sector.index
    df = genes_df %>% filter(chr == chr_name)
    
    if (nrow(df) > 0) {
      # Draw mean baseline using circos.lines (FIX for circos.abline error)
      circos.lines(x = CELL_META$xlim, y = c(mean_gene, mean_gene), 
                   col = "grey60", lty = 2, lwd = 0.5)
      
      # Plot the density as a smooth line
      circos.lines(x = df$mid_bp, y = df$density, 
                   col = GENE_COLOR, lwd = 1.5)
    }
    
    if(CELL_META$sector.index == chr_len_df$chr[1]) {
      circos.text(CELL_META$xlim[1], 0.5, "Gene Density", niceFacing = TRUE,
                  adj = c(-0.1, 0), cex = 0.7, col = GENE_COLOR)
    }
  }
)

# Track 2: LTR/Copia Density (Using Lines)
circos.track(
  ylim = c(0, 1),
  track.height = 0.08,
  bg.col = "white",
  bg.border = "grey50",
  panel.fun = function(x, y) {
    chr_name = CELL_META$sector.index
    df = copia_df %>% filter(chr == chr_name)
    
    if (nrow(df) > 0) {
      # Draw mean baseline using circos.lines (FIX for circos.abline error)
      circos.lines(x = CELL_META$xlim, y = c(mean_copia, mean_copia), 
                   col = "grey60", lty = 2, lwd = 0.5)
      
      # Plot the density as a smooth line
      circos.lines(x = df$mid_bp, y = df$density, 
                   col = COPIA_COLOR, lwd = 1.5)
    }
    
    if(CELL_META$sector.index == chr_len_df$chr[1]) {
      circos.text(CELL_META$xlim[1], 0.5, "LTR/Copia", niceFacing = TRUE,
                  adj = c(-0.1, 0), cex = 0.7, col = COPIA_COLOR)
    }
  }
)

# Track 3: LTR/Gypsy Density (Using Lines)
circos.track(
  ylim = c(0, 1),
  track.height = 0.08,
  bg.col = "white",
  bg.border = "grey50",
  panel.fun = function(x, y) {
    chr_name = CELL_META$sector.index
    df = gypsy_df %>% filter(chr == chr_name)
    
    if (nrow(df) > 0) {
      # Draw mean baseline using circos.lines (FIX for circos.abline error)
      circos.lines(x = CELL_META$xlim, y = c(mean_gypsy, mean_gypsy), 
                   col = "grey60", lty = 2, lwd = 0.5)
      
      # Plot the density as a smooth line
      circos.lines(x = df$mid_bp, y = df$density, 
                   col = GYPSY_COLOR, lwd = 1.5)
    }
    
    if(CELL_META$sector.index == chr_len_df$chr[1]) {
      circos.text(CELL_META$xlim[1], 0.5, "LTR/Gypsy", niceFacing = TRUE,
                  adj = c(-0.1, 0), cex = 0.7, col = GYPSY_COLOR)
    }
  }
)

# Track 4: Satellite DNA Density (Using Lines)
circos.track(
  ylim = c(0, 1),
  track.height = 0.08,
  bg.col = "white",
  bg.border = "grey50",
  panel.fun = function(x, y) {
    chr_name = CELL_META$sector.index
    df = satellite_df %>% filter(chr == chr_name)
    
    if (nrow(df) > 0) {
      # Draw mean baseline using circos.lines (FIX for circos.abline error)
      circos.lines(x = CELL_META$xlim, y = c(mean_satellite, mean_satellite), 
                   col = "grey60", lty = 2, lwd = 0.5)
      
      # Plot the density as a smooth line
      circos.lines(x = df$mid_bp, y = df$density, 
                   col = SATELLITE_COLOR, lwd = 1.5)
    }
    
    if(CELL_META$sector.index == chr_len_df$chr[1]) {
      circos.text(CELL_META$xlim[1], 0.5, "Satellite DNA", niceFacing = TRUE,
                  adj = c(-0.1, 0), cex = 0.7, col = SATELLITE_COLOR)
    }
  }
)

# Track 5 (Outermost Data Track): DNA Transposon Density (Using Lines)
circos.track(
  ylim = c(0, 1),
  track.height = 0.08,
  bg.col = "white",
  bg.border = "grey50",
  panel.fun = function(x, y) {
    chr_name = CELL_META$sector.index
    df = dna_df %>% filter(chr == chr_name)

    if (nrow(df) > 0) {
      # Draw mean baseline using circos.lines (FIX for circos.abline error)
      circos.lines(x = CELL_META$xlim, y = c(mean_dna, mean_dna), 
                   col = "grey60", lty = 2, lwd = 0.5)
      
      # Plot the density as a smooth line
      circos.lines(x = df$mid_bp, y = df$density, 
                   col = DNA_COLOR, lwd = 1.5)
    }

    if(CELL_META$sector.index == chr_len_df$chr[1]) {
      circos.text(CELL_META$xlim[1], 0.5, "DNA Transposons", niceFacing = TRUE,
                  adj = c(-0.1, 0), cex = 0.7, col = DNA_COLOR)
    }
  }
)


# Close the PDF device to save the plot
dev.off()

cat("\nSUCCESS: 5-track Circos plot saved to carex_borbonica_circos_plot_line.pdf (using lines for smoother visualization)\n")

