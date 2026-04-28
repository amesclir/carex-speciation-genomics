#!/usr/bin/bash
#SBATCH --job-name=A0_downloading
#SBATCH --output=A0_downloading_output
#SBATCH --error=A0_downloading_error
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=40G
#SBATCH --partition=standard
#SBATCH --time=12:00:00



wget -O /home/aescudero/genome_assembly/raw_data/PacBio/E_quinqueflora/E_quinqueflora_hifi_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR120/005/ERR12085105/ERR12085105.fastq.gz

wget -O /home/aescudero/genome_assembly/raw_data/hiC/E_quinqueflora/E_quinqueflora_hiC_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR121/088/ERR12102388/ERR12102388_1.fastq.gz

wget -O /home/aescudero/genome_assembly/raw_data/hiC/E_quinqueflora/E_quinqueflora_hiC_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR121/088/ERR12102388/ERR12102388_2.fastq.gz
