# Save this content into a file named 'check_chars.sh'
#!/bin/bash

ALIGNMENT_FILE="/home/aescudero/wgg/bams/all_samples_polymorphic_snps_mac20.min4.phy.varsites.phy.fixed.phy"

echo "Checking for non-IUPAC DNA characters exclusively in the sequence data of: $ALIGNMENT_FILE"
echo "--------------------------------------------------------------------------------------"

if [ ! -f "$ALIGNMENT_FILE" ]; then
    echo "ERROR: Alignment file not found: $ALIGNMENT_FILE. Exiting."
    exit 1
fi

tail -n +2 "$ALIGNMENT_FILE" | \
awk '{
    name = $1;
    sequence = substr($0, length(name) + 2);
    gsub(" ", "", sequence);
    print sequence;
}' | \
grep -io '[^ACGT_RY_S_W_K_M_B_D_H_V_N\-]' | \
sort | uniq -c | sort -nr
