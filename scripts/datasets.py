import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


class GraphDataset(Dataset):
    def __init__(self, names_file, feats_dir, feats_type='embeddings', edges_type='normal'):
        # Initialize data
        self.names = list(np.loadtxt(names_file, dtype=str))
        self.feats_dir = feats_dir
        self.feats_type = feats_type
        self.edges_type = edges_type

        if self.feats_type == 'onehot':
            # Init dictionary for one-hot encoding
            alphabet = 'ARNDCQEGHILKMFPSTWYVUOBZJX'
            self.ohdict = dict((c, i) for i, c in enumerate(alphabet))

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]

        # Load pickle file with dictionary containing embeddings (LxF), edge_indexes and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)

        # Select features type
        if self.feats_type == 'embeddings':
            features = d['embeddings']

        elif self.feats_type == 'onehot':
            features = np.zeros((seqlen, len(self.ohdict)), dtype=np.float32)
            for i in range(seqlen):
                features[i, self.ohdict[seq[i]]] = 1

        elif self.feats_type == 'nofeats':
            features = np.ones((seqlen, 1), dtype=np.float32)

        elif self.feats_type == 'degree':
            e = d['edges'][0]
            _, cn = np.unique(e, return_counts=True)
            cn -= 1
            features = cn.reshape(-1, 1).astype(np.float32)

        else:
            print('[!] Unknown features type, try "embeddings", "onehot", "degree" or "nofeats".')
            exit(0)

        # Select edges type
        if self.edges_type == 'normal':
            edges, _ = remove_self_loops(np.array(d['edges']))   # remove elements in diagonal (self-loops)

        elif self.edges_type == 'random':
            d2 = pickle.load(open(self.feats_dir + '/modif_edges/' + name + '.pkl', 'rb'))
            edges = np.array(d2['edges_rand'])

        elif self.edges_type == 'identity':
            edges = np.array(np.diag_indices(seqlen))

        else:
            print('[!] Unknown edges type, try "normal", "random", "identity".')
            exit(0)

        # Get labels
        labels = d['labels'].toarray().astype(np.float32)

        # Include pseudo (for GMM)
        U = np.zeros((edges.shape[1],))
        for i, e in enumerate(edges.T):
            U[i] = e[0] - e[1]

        return Data(x=torch.from_numpy(features),
                    edge_index=torch.from_numpy(edges),
                    y=torch.from_numpy(labels),
                    pseudo=torch.from_numpy(U))



class CNN1DDataset(Dataset):
    def __init__(self, names_file, feats_dir, feats_type='embeddings'):
        # Initialize data
        self.names = list(np.loadtxt(names_file, dtype=str))
        self.feats_dir = feats_dir
        self.feats_type = feats_type

        if self.feats_type == 'onehot':
            # Init dictionary for one-hot encoding
            alphabet = 'ARNDCQEGHILKMFPSTWYVUOBZJX'
            self.ohdict = dict((c, i) for i, c in enumerate(alphabet))

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]

        # Load pickle file with dictionary containing embeddings (LxF), sequence (L) and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)

        # Select features type
        if self.feats_type == 'embeddings':
            features = d['embeddings'].T

        elif self.feats_type == 'onehot':
            features = np.zeros((len(self.ohdict), seqlen), dtype=np.float32)
            for i in range(seqlen):
                features[self.ohdict[seq[i]], i] = 1

        else:
            print('[!] Unknown features type, try "embeddings" or "onehot".')
            exit(0)

        # Get labels (N)
        labels = d['labels'].toarray().astype(np.float32).squeeze()

        return features, seqlen, labels



class CNN2DDataset(Dataset):
    def __init__(self, data_file, feats_dir):
        # Initialize data
        data = np.loadtxt(data_file, dtype=str)
        try:
            self.names = data[:,0]
            self.lengths = data[:,1].astype(np.int)
        except:
            self.names = data

        self.feats_dir = feats_dir

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def get_one(self, name):
        # Load pickle file with dictionary containing sequence (L) and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seqlen = len(d['sequence'])

        # Get contact map (LxL)
        cmap = np.zeros((seqlen, seqlen), dtype=np.float32)
        cmap[d['edges']] = 1
        cmap = cmap - np.identity(seqlen)   # delete diagonal (adjacency matrix)

        # Get labels (N)
        labels = d['labels'].toarray().astype(np.float32).squeeze()

        return cmap, seqlen, labels

    def __getitem__(self, index):
        # Load batch of samples
        batch_names = self.names[np.squeeze(index)]

        try:
            batch_cmaps, batch_lengths, batch_labels = self.get_one(batch_names)

        except:
            batch_cmaps = []
            batch_lengths = []
            batch_labels = []
            for name in batch_names:
                cmap, seqlen, labels = self.get_one(name)
                batch_cmaps.append(cmap)
                batch_lengths.append(seqlen)
                batch_labels.append(labels)

        return batch_cmaps, batch_lengths, batch_labels



class CNN1D2DDataset(Dataset):
    def __init__(self, data_file, feats_dir, feats_type='embeddings'):
        # Initialize data
        data = np.loadtxt(data_file, dtype=str)
        try:
            self.names = data[:,0]
            self.lengths = data[:,1].astype(np.int)
        except:
            self.names = data

        self.feats_dir = feats_dir
        self.feats_type = feats_type

        if self.feats_type == 'onehot':
            # Init dictionary for one-hot encoding
            alphabet = 'ARNDCQEGHILKMFPSTWYVUOBZJX'
            self.ohdict = dict((c, i) for i, c in enumerate(alphabet))

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def get_one(self, name):
        # Load pickle file with dictionary containing embeddings (LxF), sequence (L) and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)

        # Get contact map (LxL)
        cmap = np.zeros((seqlen, seqlen), dtype=np.float32)
        cmap[d['edges']] = 1
        cmap = cmap - np.identity(seqlen)   # delete diagonal (adjacency matrix)

        # Select features type
        if self.feats_type == 'embeddings':
            embeddings = d['embeddings'].T

        elif self.feats_type == 'onehot':
            embeddings = np.zeros((len(self.ohdict), seqlen), dtype=np.float32)
            for i in range(seqlen):
                embeddings[self.ohdict[seq[i]], i] = 1

        else:
            print('[!] Unknown features type, try "embeddings" or "onehot".')
            exit(0)

        # Get labels (N)
        labels = d['labels'].toarray().astype(np.float32).squeeze()

        return cmap, embeddings, seqlen, labels

    def __getitem__(self, index):
        # Load batch of samples
        batch_names = self.names[np.squeeze(index)]

        try:
            batch_cmaps, batch_embeddings, batch_lengths, batch_labels = self.get_one(batch_names)

        except:
            batch_cmaps = []
            batch_embeddings = []
            batch_lengths = []
            batch_labels = []
            for name in batch_names:
                cmap, embeddings, seqlen, labels = self.get_one(name)
                batch_cmaps.append(cmap)
                batch_embeddings.append(embeddings)
                batch_lengths.append(seqlen)
                batch_labels.append(labels)

        return batch_cmaps, batch_embeddings, batch_lengths, batch_labels



class MLPDataset(Dataset):
    def __init__(self, names_file, feats_dir, feats_type='embeddings'):
        # Initialize data
        self.names = list(np.loadtxt(names_file, dtype=str))
        self.feats_dir = feats_dir
        self.feats_type = feats_type

        if self.feats_type == 'onehot':
            # Init dictionary for one-hot encoding
            alphabet = 'ARNDCQEGHILKMFPSTWYVUOBZJX'
            self.ohdict = dict((c, i) for i, c in enumerate(alphabet))

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]

        # Load pickle file with dictionary containing embeddings (LxF), sequence (L) and labels (1xN)
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)

        # Select features type
        if self.feats_type == 'embeddings':
            features = d['embeddings'].T

        elif self.feats_type == 'onehot':
            features = np.zeros((len(self.ohdict), seqlen), dtype=np.float32)
            for i in range(seqlen):
                features[self.ohdict[seq[i]], i] = 1

        else:
            print('[!] Unknown features type, try "embeddings" or "onehot".')
            exit(0)

        # Get protein-level features
        features = np.mean(features, 1)

        # Get labels (N)
        labels = d['labels'].toarray().astype(np.float32).squeeze()

        return features, labels
