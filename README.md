# Carex Speciation Genomics

**Repository for the genomic analyses accompanying the manuscript:** *"Chromosomal inversions accelerate genetic evolution and drive ecological speciation across an island gradient"* **Authors:** Ines Gómez-Ramos, Rogelio Sánchez-Villegas, Ashwini V. Mohan, Christophe Lavergne, José Cerca, José I. Márquez-Corro, André Marques, Santiago Martín-Bravo, Modesto Luceño, Kay Lucek, Marcial Escudero.

## Overview
This repository contains the scripts, pipelines, and code used to analyze the rapid ecological speciation between two sister *Carex* species (*C. boryana* and *C. borbonica*) on Réunion Island. The workflow includes *de novo* chromosome-level genome assembly, structural variant (SV) identification, whole-genome population genetics, introgression analysis, landscape genomics (RDA), and tests for purifying selection.

---

## Repository Structure & Workflow
The repository is organized into directories corresponding to the major analytical steps described in the manuscript.

### 1. `genome_assembly/`
Scripts for *de novo* assembly, scaffolding, decontamination, and annotation of the *C. boryana* and *C. borbonica* reference genomes.
* **`raw_data/`**: QC of raw PacBio reads (Jellyfish, Nanoplot, Smudgeplot).
* **`A1` - `A3`**: Initial HiFi/Hi-C assembly (hifiasm), haplotig purging (purge_dups), and assembly QC (Merqury, Compleasm).
* **`B1` - `B8`**: Hi-C mapping, chromosome-level scaffolding (Juicer/YaHS), decontamination (Blobtools/Tiara), and reciprocal cross-species Hi-C mapping to validate structural variants.
* **`C1` - `C3`**: Repeat masking (RepeatMasker/EarlGrey) and genome annotation (BRAKER3).
* Python/R scripts for final genome visualization.

### 2. `syri/` & `GENESPACE/`
Identification and validation of structural rearrangements (inversions, translocations, duplications).
* **`syri/`**: Scripts for running minimap2 and SyRI to identify genome-wide structural variants, followed by visualization with plotsr.
* **`GENESPACE/`**: Synteny and collinearity tracking between the two assemblies (`GENESPACE.Rmd`).

### 3. `wgg/` (Whole Genome Genotyping & Population Genomics)
Pipeline for processing short-read sequencing data (184 individuals), variant calling, and population-level analyses.
* **Core variant calling**: Scripts for BCFtools mpileup/call and VCF filtering (`wgg_bcftools_call.sbatch`, `wgg_vcf_filter.sbatch`).
* **`admixture_data/`**: Scripts for running ADMIXTURE on the global background and specific SV compartments, including LD pruning and plotting scripts.
* **`iqtree/`**: Maximum likelihood phylogenetic inference (`run_wgg_iqtree.sbatch`).
* **`dsuite/`**: Scripts for calculating Patterson’s D and *fd* statistics to test for introgression across the genomic background and within focal inversions (ABBA-BABA).
* **`fst_results/` & `ld_decay/`**: Scripts for calculating windowed and per-SNP $F_{ST}$, nucleotide diversity ($\pi$), $d_{XY}$, and matched Linkage Disequilibrium (LD) decay inside vs. outside structural variants.
* **`RDA/`**: Landscape genomics. Includes scripts for downloading WorldClim data, selecting independent variables, and running standard and partial Redundancy Analyses (pRDA) to test for environmental adaptation independent of population structure.

### 4. `purifying_selection/`
Custom pipeline to calculate evolutionary rates ($\omega=dN/dS$) and neutral substitution rates (P4D) inside vs. outside structural variants to test for Hill-Robertson interference / relaxed purifying selection.
* **`run1` - `run3`**: Extraction of genes located strictly within vs. outside inversions and CDS sequence parsing.
* **`run4` - `run6`**: Ortholog identification, protein alignment (MAFFT), and back-translation to codon alignments (PAL2NAL).
* **`run7` - `run9`**: Preparation and execution of PAML (codeml), and parsing of final $dN/dS$ and 4-fold degenerate site outputs.

### 5. `GO_enrichment/` & `maps/`
Scripts and markdown files for the functional annotation of candidate genes and geographical visualization of sampling sites.
* **`GO_enrichment/`**: Script (`go_annotation.Rmd`) for Gene Ontology (GO) enrichment analysis and semantic clustering (GOSemSim) of candidate adaptive genes located within focal inversions.
* **`maps/`**: R markdown for plotting geographical sampling locations across the climatic gradient of Réunion Island.
