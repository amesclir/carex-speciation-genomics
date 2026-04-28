import PyPDF2
import os

def combine_pdfs_to_single_figure(input_files, output_filename="final_combined_genomic_figure.pdf"):
    """
    Merges multiple PDF files into a single output PDF.
    This is useful for creating multi-panel figures for publications.
    
    NOTE: The 'PyPDF2' library must be installed (pip install PyPDF2).
    """
    pdf_merger = PyPDF2.PdfMerger()
    
    print("Starting PDF merger...")
    
    # List of files to merge
    valid_files = [f for f in input_files if os.path.exists(f)]
    missing_files = [f for f in input_files if not os.path.exists(f)]

    if not valid_files:
        print("Error: No PDF files were found to merge. Please check your input paths.")
        return
        
    for filename in valid_files:
        try:
            # Append the page from the current PDF to the merger
            pdf_merger.append(filename)
            print(f"Successfully added: {filename}")
        except Exception as e:
            print(f"Could not read or add {filename}. Error: {e}")

    # Write the merged PDF to the output file
    with open(output_filename, 'wb') as outfile:
        pdf_merger.write(outfile)

    pdf_merger.close()

    print("\n--- Summary ---")
    if missing_files:
        print(f"Warning: The following files were skipped because they were not found: {missing_files}")
    print(f"Successfully merged {len(valid_files)} files into: {output_filename}")


# --- 1. Define the list of files to be merged ---
# This list is based on the output names from your Python scripts.
input_pdfs = [
    "multipanel_genomic_plot_chr14.pdf",
    "multipanel_genomic_plot_chr18.pdf",
    "multipanel_genomic_plot_chr20.pdf",
    "multipanel_genomic_plot_chr3.pdf",
    "multipanel_genomic_plot_chr32.pdf",
    "multipanel_genomic_plot_chr28.pdf",
]

# --- 2. Execute the merger function ---
combine_pdfs_to_single_figure(input_pdfs)
