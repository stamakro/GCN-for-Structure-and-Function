import sys
import numpy as np
import pickle
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def calculate_angle(p1, p2, p3):
    # Calculate q vectors
    q1 = p2 - p1
    q2 = p2 - p3

    # Calculate cosine and angle
    cos_angle = np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
    angle = np.arccos(cos_angle)
    return angle

def calculate_dihedral(p1, p2, p3, p4):
    # Calculate q vectors
    q1 = p2 - p1
    q2 = p3 - p2
    q3 = p4 - p3

    # Calculate cross vectors
    q1_x_q2 = np.cross(q1, q2)
    q2_x_q3 = np.cross(q2, q3)

    # Calculate normal vectors
    n1 = q1_x_q2 / np.sqrt(np.dot(q1_x_q2, q1_x_q2))
    n2 = q2_x_q3 / np.sqrt(np.dot(q2_x_q3, q2_x_q3))

    # Calculate unit vectors
    u1 = n2
    u3 = q2 / (np.sqrt(np.dot(q2, q2)))
    u2 = np.cross(u3, u1)

    # Calculate cosine and sine
    cos_angle = np.dot(n1, u1)
    sin_angle = np.dot(n1, u2)

    # Calculate angle
    angle = - np.arctan2(sin_angle, cos_angle)
    return angle


def main(pdb_file, dssp_exe, out_file):
    # Read PDB structrue
    p = PDBParser()
    structure = p.get_structure('id', pdb_file)
    model = structure[0]

    # Run DSSP
    dssp = DSSP(model, pdb_file, dssp=dssp_exe)
    # keys:
    # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
    sec_struc = []
    rel_asa = []
    for a_key in list(dssp.keys()):
        aux = dssp[a_key]
        sec_struc.append(aux[2])
        rel_asa.append(0 if aux[3] == 'NA' else aux[3])
    rel_asa = np.expand_dims(rel_asa, axis=1)

    # Get coordinates for N, CA and C atoms
    coor_N = []
    coor_CA = []
    coor_C = []
    for chain in model.get_list():
        total_len = len(chain.get_list())
        for residue in chain.get_list():
            coor_N.append(residue['N'].get_coord())
            coor_CA.append(residue['CA'].get_coord())
            coor_C.append(residue['C'].get_coord())

    assert len(coor_N) == len(coor_CA) == len(coor_C) == total_len

    # Get dihedral angles and inter-residue angles
    angles = []
    for j in range(total_len):
        # Initialize angles if not found
        phi_angle, psi_angle, omega_angle = (2*np.pi, 2*np.pi, 2*np.pi)
        theta_angle, tau_angle = (2*np.pi, 2*np.pi)

        vec_N2, vec_CA2, vec_C2 = (coor_N[j], coor_CA[j], coor_C[j])

        # Phi
        if j != 0:
            vec_C1 = coor_C[j-1]
            phi_angle = calculate_dihedral(vec_C1, vec_N2, vec_CA2, vec_C2)

        # Psi and Omega
        if j != total_len-1:
            vec_N3 = coor_N[j+1]
            psi_angle = calculate_dihedral(vec_N2, vec_CA2, vec_C2, vec_N3)
            vec_CA3 = coor_CA[j+1]
            omega_angle = calculate_dihedral(vec_CA2, vec_C2, vec_N3, vec_CA3)

        # Theta
        if np.logical_and(j != 0, j != total_len-1):
            vec_CA1 = coor_CA[j-1]
            vec_CA3 = coor_CA[j+1]
            theta_angle = calculate_angle(vec_CA1, vec_CA2, vec_CA3)

        # Tau
        if np.logical_and(j > 1, j != total_len-1):
            vec_CA0, vec_CA1, vec_CA3 = (coor_CA[j-2], coor_CA[j-1], coor_CA[j+1])
            tau_angle = calculate_dihedral(vec_CA0, vec_CA1, vec_CA2, vec_CA3)

        # Concatenate angles
        angles.append([phi_angle, psi_angle, theta_angle, tau_angle])

    # Calculate sine and cosine of each angle
    angles = np.array(angles)
    angles_sin = np.sin(angles)
    angles_cos = np.cos(angles)
    angles_sin_cos = np.stack([angles_sin[:, 0], angles_cos[:, 0],
                                angles_sin[:, 1], angles_cos[:, 1],
                                angles_sin[:, 2], angles_cos[:, 2],
                                angles_sin[:, 3], angles_cos[:, 3]]).T
    angles_sin_cos[np.where(np.abs(angles_sin_cos) < 1e-10)] = 0

    # One-hot encoding of secondary structure (8-state)
    alphabet = 'HBEGITS-' # H Alpha helix (4-12)
                            # B Isolated beta-bridge residue
                            # E Strand
                            # G 3-10 helix
                            # I Pi helix
                            # T Turn
                            # S Bend
                            # - None
    ohdict = dict((c, i) for i, c in enumerate(alphabet))
    ss_onehot = np.zeros((total_len, len(ohdict)))
    for i in range(total_len):
        ss_onehot[i, ohdict[sec_struc[i]]] = 1

    # Create feature matrix for the protein
    features = np.hstack([ss_onehot, rel_asa, angles_sin_cos])

    # Save dictionary of features
    with open(out_file, 'wb') as f:
        pickle.dump(features.astype('float32'), f, protocol=2)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: %s <pdb_file, dssp_exe, out_file>' % sys.argv[0])
    pdb_file, dssp_exe, out_file = sys.argv[1:]

    main(pdb_file, dssp_exe, out_file)