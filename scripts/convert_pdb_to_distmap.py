import sys
import numpy as np
from Bio.PDB import PDBParser


def calculate_distance(residues):
    dist = []
    for res1 in residues:
        if 'CA' not in res1:
            continue
        c1 = res1['CB'] if 'CB' in res1 else res1['CA']
        tmp_dist = []
        for res2 in residues:
            if 'CA' not in res2:
                continue
            c2 = res2['CB'] if 'CB' in res2 else res2['CA']
            tmp_dist.append(c1 - c2)
        dist.append(tmp_dist)
    return np.matrix(dist)


pdb_file, distmap_file = sys.argv[1:]


p = PDBParser(PERMISSIVE=1)
structure = p.get_structure('', pdb_file)
model = structure[0]
residues = []
for chain in model:
    for residue in chain:
        residues.append(residue)

dist = calculate_distance(residues)
np.save(distmap_file, dist)
