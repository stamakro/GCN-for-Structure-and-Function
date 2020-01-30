import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Sampler


class CustomData(Data):
    def __init__(self, x=None, mask=None, y=None, **kwargs):
        super(CustomData, self).__init__()

        self.x = x
        self.mask = mask
        self.y = y
        
        for key, item in kwargs.items():
            self[key] = item



def cnn1d_collate(batch):
    # Get data, label and length (from a list of arrays)
    feats = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Pad data to max sequence length in batch
    max_len = max(lengths)
    feats_pad = [np.pad(item, ((0,0),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]

    # Create mask for each sample
    masks = [np.expand_dims([1] * i + [0] * (max_len - i), axis=0) for i in lengths]
    
    return CustomData(x=torch.from_numpy(np.array(feats_pad)),
                      mask=torch.from_numpy(np.array(masks, dtype=np.float32)),
                      y=torch.from_numpy(np.array(labels)))



def cnn2d_collate(batch):
    # Get data, length and label (from a list of arrays)
    feats = batch[0][0]
    lengths = batch[0][1]
    labels = batch[0][2]

    try:
        # Pad data to max sequence length in batch and create mask for each sample
        max_len = max(lengths)
        feats_pad = [np.pad(item, ((0,max_len-lengths[i]),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]
        masks = [np.pad(np.ones((i, i)), ((0,max_len-i),(0,max_len-i)), 'constant') for i in lengths]   
    except:
        feats_pad = np.expand_dims(feats, axis=0)
        masks = np.ones((1, lengths, lengths))
    
    return CustomData(x=torch.from_numpy(np.expand_dims(np.array(feats_pad, np.float32), axis=1)),
                      mask=torch.from_numpy(np.expand_dims(np.array(masks, np.float32), axis=1)),
                      y=torch.from_numpy(np.array(labels)))



def cnn1d2d_collate(batch):
    # Get data, length and label (from a list of arrays)
    feats = batch[0][0]
    embeddings = batch[0][1]
    lengths = batch[0][2]
    labels = batch[0][3]

    try:
        # Pad data to max sequence length in batch and create mask for each sample
        max_len = max(lengths)
        feats_pad = [np.pad(item, ((0,max_len-lengths[i]),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]
        masks = [np.pad(np.ones((i, i)), ((0,max_len-i),(0,max_len-i)), 'constant') for i in lengths]
        
        embeddings = [np.pad(item, ((0,0),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(embeddings)]
        masks1d = [[1] * i + [0] * (max_len - i) for i in lengths]
        
    except:
        feats_pad = np.expand_dims(feats, axis=0)
        masks = np.ones((1, lengths, lengths))
        
        embeddings = np.expand_dims(embeddings, axis=0)
        masks1d = np.ones((1, lengths))
    
    return CustomData(x=torch.from_numpy(np.expand_dims(np.array(feats_pad, np.float32), axis=1)),
                      mask=torch.from_numpy(np.expand_dims(np.array(masks, np.float32), axis=1)),
                      y=torch.from_numpy(np.array(labels)),
                      emb=torch.from_numpy(np.array(embeddings)),
                      mask1d=torch.from_numpy(np.expand_dims(np.array(masks1d, np.float32), axis=1)))



def mlp_collate(batch):
    # Get data, label and length (from a list of arrays)
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    return CustomData(x=torch.from_numpy(np.array(feats)), y=torch.from_numpy(np.array(labels)))



class VariableBatchSizeSampler(Sampler):
    def __init__(self, data_set, batch_sizes=[64, 32, 16, 8, 4, 1, 1, 1, 1, 1], sep=100, shuffle=False, drop_last=False):
        self.lengths = data_set.lengths
        self.num_samples = len(data_set)
        
        self.batch_sizes = batch_sizes
        self.cluster_types = np.arange(len(batch_sizes))
        self.lengths_cluster = (self.lengths / sep).astype(np.int)
        
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        
        batch_lists = []
        for j in self.cluster_types:
            
            # get indexes for each cluster type and shuffle
            idx = np.argwhere(self.lengths_cluster == j).tolist()
            if self.shuffle:
                np.random.shuffle(idx)
            
            # create batches
            batches = [idx[i:i+self.batch_sizes[j]] for i in range(0, len(idx), self.batch_sizes[j])]
            
            # filter out the shorter batches
            if self.drop_last:
                batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]

            batch_lists.append(batches)
        
        # flatten lists and shuffle the batches
        lst = [item for sublist in batch_lists for item in sublist]
        if self.shuffle:
            np.random.shuffle(lst)
        
        return iter(lst)
    
    def __len__(self):
        return self.num_samples
