#!/bin/sh


PROTEINID=1ax8A

# Extract ELMo embeddings
FASTAFILE=example/${PROTEINID}.fasta
EMBFILE=example/${PROTEINID}.embeddings.pkl

python scripts/seqvec_embedder.py --input=${FASTAFILE} --output=${EMBFILE} --model=test/uniref50_v2


# Get distance map from PDB structure
PDBFILE=example/${PROTEINID}.pdb
DMAPFILE=example/${PROTEINID}.distmap.npy

python scripts/convert_pdb_to_distmap.py ${PDBFILE} ${DMAPFILE}


# Create dictionary with all needed features
LABELSFILE=data_pdb/Yterms.pkl
OUTFILE=example/${PROTEINID}.pkl

python scripts/generate_feats.py ${PROTEINID} ${FASTAFILE} ${EMBFILE} ${DMAPFILE} ${LABELSFILE} ${OUTFILE}
