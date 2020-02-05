import argparse
import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader as pyDataLoader
from torch_geometric.data import DataLoader as pygeoDataLoader
from datasets import GraphDataset, CNN1DDataset, CNN2DDataset, CNN1D2DDataset, MLPDataset
from dataset_utils import cnn1d_collate, cnn2d_collate, cnn1d2d_collate, mlp_collate
from dataset_utils import VariableBatchSizeSampler

from networks import GraphCNN, CNN1D_dilated, CNN2D, CNN1D2D, Perceptron
from model import train, test, extract

def str2bool(value):
    return value.lower() == 'true'


parser = argparse.ArgumentParser(description='')
# Learning hyperparameters
parser.add_argument('--phase',          dest='phase',           default='train') # 'train' / 'test' / 'extract'
parser.add_argument('--batch_size',     dest='batch_size',      type=int,       default=64)
parser.add_argument('--num_epochs',     dest='num_epochs',      type=int,       default=80)
parser.add_argument('--init_lr',        dest='init_lr',         type=float,     default=0.0005)
parser.add_argument('--lr_sched',       dest='lr_sched',        type=str2bool,  default=True)

# Architecture hyperparameters
parser.add_argument('--net_type',       dest='net_type',        default='gcn')  # 'gcn' / 'chebcn' / 'gmmcn' / 'gincn' / 'cnn1d' / 'cnn2d' / 'cnn1d2d' / 'mlp'
parser.add_argument('--feats_type',     dest='feats_type',      default='embeddings') # 'embeddings' / 'onehot' / 'nofeats'
parser.add_argument('--edges_type',     dest='edges_type',      default='normal') # 'normal' / 'random' / 'identity'
parser.add_argument('--input_dim',      dest='input_dim',       type=int,       default=1024)
parser.add_argument('--channel_dims',   dest='channel_dims',    type=str,       default='256_256_512')
parser.add_argument('--filter_sizes',   dest='filter_sizes',    type=str,       default='5_5') # for 'cnn1d' and 'cnn2d'
parser.add_argument('--fc_dim',         dest='fc_dim',          type=str,       default='512')
parser.add_argument('--num_classes',    dest='num_classes',     type=int,       default=256)
parser.add_argument('--cheb_order',     dest='cheb_order',      type=int,       default=2) # for 'chebcn'

#input format
parser.add_argument('--protein_level',     dest='protein_level',      type=int,       default=0) # if true the embeddings are expected to be given as protein-level, else aa-level

# Training directories and files
parser.add_argument('--model_dir',      dest='model_dir',       default='models/GCN')
parser.add_argument('--train_file',     dest='train_file',      default='data/train.names')
parser.add_argument('--valid_file',     dest='valid_file',      default='data/valid.names')
parser.add_argument('--feats_dir',      dest='feats_dir',       default='feats')
parser.add_argument('--icvec_file',     dest='icvec_file',      default='data/icVec.npy')

# Test directories and files
parser.add_argument('--model_file',     dest='model_file',      default='models/GCN/checkpoint/model_epoch40.pth.tar')
parser.add_argument('--test_file',      dest='test_file',       default='data/test.names')
parser.add_argument('--save_file',      dest='save_file',       default='models/GCN/test_pred_ep40.pkl')
parser.add_argument('--emb_save_file',  dest='emb_save_file',   default='models/GCN/embeddings.pkl')
args = parser.parse_args()


# Check cuda availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("[*] Selected device: ", device)
print("[*] Using %s features." % args.feats_type)

# Initialize network model
if args.net_type == 'gcn' or args.net_type == 'chebcn' or args.net_type == 'gmmcn' or args.net_type == 'gincn':
    net = GraphCNN(input_dim=args.input_dim, net_type=args.net_type,
                   channel_dims=list(map(int, args.channel_dims.split("_"))),
                   fc_dim=args.fc_dim, num_classes=args.num_classes, cheb_order=args.cheb_order).to(device)

elif args.net_type == 'cnn1d':
    net = CNN1D_dilated(input_dim=args.input_dim, num_filters=list(map(int, args.channel_dims.split("_"))),
                        filter_sizes=list(map(int, args.filter_sizes.split("_"))),
                        fc_dim=args.fc_dim, num_classes=args.num_classes).to(device)

elif args.net_type == 'cnn2d':
    net = CNN2D(input_dim=1, num_filters=list(map(int, args.channel_dims.split("_"))),
                filter_sizes=list(map(int, args.filter_sizes.split("_"))),
                fc_dim=args.fc_dim, num_classes=args.num_classes).to(device)

elif args.net_type == 'cnn1d2d':
    net = CNN1D2D(input_dim_1d=args.input_dim, input_dim_2d=1,
                  num_filters_1d=list(map(int, args.channel_dims.split("_"))),
                  filter_sizes_1d=list(map(int, args.filter_sizes.split("_"))),
                  num_filters_2d=list(map(int, args.channel_dims.split("_"))),
                  filter_sizes_2d=list(map(int, args.filter_sizes.split("_"))),
                  fc_dim=args.fc_dim, num_classes=args.num_classes).to(device)

elif args.net_type == 'mlp':
    net = Perceptron(input_dim=args.input_dim, fc_dim=list(map(int, args.fc_dim.split("_"))), num_classes=args.num_classes).to(device)

else:
    print('[!] Unknown network type, try "gcn", "chebcn", "gmmcn", "gincn", "cnn1d", "cnn2d", "cnn1d2d" or "mlp".')
    exit(0)

print("[*] Initialize model successfully, network type: %s." % args.net_type)
print(net)
print("[*] Number of model parameters:")
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

# Load GO-term IC vector
icvec = np.load(args.icvec_file).astype(np.float32)
assert icvec.size == args.num_classes

# Define loss function
criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)


# Training phase
if args.phase == 'train':

    # Create directories for checkpoint, sample and logs files
    ckpt_dir = args.model_dir + '/checkpoint'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logs_dir = args.model_dir + '/logs'

    # Data loading
    print("[*] Loading training and validation data...")
    if args.net_type == 'gcn' or args.net_type == 'chebcn' or args.net_type == 'gmmcn' or args.net_type == 'gincn':
        train_set = GraphDataset(args.train_file, args.feats_dir, args.feats_type, args.edges_type)
        train_loader = pygeoDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        train_loader_eval = pygeoDataLoader(train_set, batch_size=1, shuffle=False)
        valid_set = GraphDataset(args.valid_file, args.feats_dir, args.feats_type, args.edges_type)
        valid_loader = pygeoDataLoader(valid_set, batch_size=1, shuffle=False)

    elif args.net_type == 'cnn1d':
        train_set = CNN1DDataset(args.train_file, args.feats_dir, args.feats_type)
        train_loader = pyDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=cnn1d_collate)
        train_loader_eval = pyDataLoader(train_set, batch_size=1, shuffle=False, collate_fn=cnn1d_collate)
        valid_set = CNN1DDataset(args.valid_file, args.feats_dir, args.feats_type)
        valid_loader = pyDataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=cnn1d_collate)

    elif args.net_type == 'cnn2d':
        train_set = CNN2DDataset(args.train_file, args.feats_dir)
        train_sampler = VariableBatchSizeSampler(train_set, shuffle=True, drop_last=False)
        train_loader = pyDataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=1, collate_fn=cnn2d_collate)
        train_loader_eval = pyDataLoader(train_set, batch_size=1, shuffle=False, collate_fn=cnn2d_collate)
        valid_set = CNN2DDataset(args.valid_file, args.feats_dir)
        valid_loader = pyDataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=cnn2d_collate)

    elif args.net_type == 'cnn1d2d':
        train_set = CNN1D2DDataset(args.train_file, args.feats_dir, args.feats_type)
        train_sampler = VariableBatchSizeSampler(train_set, shuffle=True, drop_last=False)
        train_loader = pyDataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=1, collate_fn=cnn1d2d_collate)
        train_loader_eval = pyDataLoader(train_set, batch_size=1, shuffle=False, collate_fn=cnn1d2d_collate)
        valid_set = CNN1D2DDataset(args.valid_file, args.feats_dir, args.feats_type)
        valid_loader = pyDataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=cnn1d2d_collate)

    elif args.net_type == 'mlp':
        train_set = MLPDataset(args.train_file, args.feats_dir, args.feats_type, args.protein_level)
        train_loader = pyDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=mlp_collate)
        train_loader_eval = pyDataLoader(train_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)
        valid_set = MLPDataset(args.valid_file, args.feats_dir, args.feats_type, args.protein_level)
        valid_loader = pyDataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)

    # Training and validation
    train(device=device, net=net, criterion=criterion,
          learning_rate=args.init_lr, lr_sched=args.lr_sched, num_epochs=args.num_epochs,
          train_loader=train_loader, train_loader_eval=train_loader_eval, valid_loader=valid_loader,
          icvec=icvec, ckpt_dir=ckpt_dir, logs_dir=logs_dir)


# Independent test phase
elif args.phase == 'test' or args.phase == 'extract':

    # Data loading
    print("\n[*] Loading test data %s." % args.test_file)
    if args.net_type == 'gcn' or args.net_type == 'chebcn' or args.net_type == 'gmmcn' or args.net_type == 'gincn':
        test_set = GraphDataset(args.test_file, args.feats_dir, args.feats_type, args.edges_type)
        test_loader = pygeoDataLoader(test_set, batch_size=1, shuffle=False)

    elif args.net_type == 'cnn1d':
        test_set = CNN1DDataset(args.test_file, args.feats_dir, args.feats_type)
        test_loader = pyDataLoader(test_set, batch_size=1, shuffle=False, collate_fn=cnn1d_collate)

    elif args.net_type == 'cnn2d':
        test_set = CNN2DDataset(args.test_file, args.feats_dir)
        test_loader = pyDataLoader(test_set, batch_size=1, shuffle=False, collate_fn=cnn2d_collate)

    elif args.net_type == 'cnn1d2d':
        test_set = CNN1D2DDataset(args.test_file, args.feats_dir, args.feats_type)
        test_loader = pyDataLoader(test_set, batch_size=1, shuffle=False, collate_fn=cnn1d2d_collate)

    elif args.net_type == 'mlp':
        test_set = MLPDataset(args.test_file, args.feats_dir, args.feats_type, args.protein_level)
        test_loader = pyDataLoader(test_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)

    # Test
    if args.phase == 'test':
        test(device=device, net=net, criterion=criterion, model_file=args.model_file,
            test_loader=test_loader, icvec=icvec, save_file=args.save_file)
    # Embedding extractor
    elif args.phase == 'extract':
        extract(device=device, net=net, model_file=args.model_file,
                names_file=args.test_file, loader=test_loader, save_file=args.emb_save_file)


else:
    print('[!] Unknown phase')
    exit(0)
