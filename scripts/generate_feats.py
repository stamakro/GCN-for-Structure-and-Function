import sys
import pickle
import numpy as np


protein_id, fasta_file, embeddings_file, distmap_file, labels_file, output_file = sys.argv[1:]


feats = dict()

# Get fasta sequence
with open(fasta_file) as fr:
    seq = fr.readlines()[1].rsplit('\n')[0]

feats['sequence'] = seq

# Get embeddings
with open(embeddings_file, 'rb') as fr:
    emb = pickle.load(fr)

feats['embeddings'] = emb[protein_id]

# Get edges (contact map from distance map)
if distmap_file is not None:
    distmap = np.load(distmap_file)
    contmap_thres = 10.
    feats['edges'] = np.where(distmap <= contmap_thres)

# Get labels
with open(labels_file, 'rb') as fr:
    Y = pickle.load(fr)

feats['labels'] = Y[protein_id]

# Sabe features file
with open(output_file, 'wb') as fw:
    pickle.dump(feats, fw)
