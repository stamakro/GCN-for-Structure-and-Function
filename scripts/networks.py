import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GMMConv, GINConv, global_add_pool


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(self, input_dim=1024, net_type='gcn', channel_dims=[256, 256, 512], fc_dim=512, num_classes=256, 
                 cheb_order=2):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [input_dim] + channel_dims
        self.net_type = net_type
        if net_type == 'gcn':
            gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True)
                          for i in range(1, len(gcn_dims))]
        elif net_type == 'chebcn':
            gcn_layers = [ChebConv(gcn_dims[i-1], gcn_dims[i], K=cheb_order, normalization='sym', bias=True)
                          for i in range(1, len(gcn_dims))]
        elif net_type == 'gmmcn':
            gcn_layers = [GMMConv(gcn_dims[i-1], gcn_dims[i], dim=1, kernel_size=1, separate_gaussians=False, bias=True)
                          for i in range(1, len(gcn_dims))]
        elif net_type == 'gincn':
            gcn_layers = [GINConv(MLP(gcn_dims[i-1], gcn_dims[i], gcn_dims[i]), eps=0, train_eps=False)
                          for i in range(1, len(gcn_dims))]
        self.gcn = nn.ModuleList(gcn_layers)

        # Define dropout
        self.drop = nn.Dropout(p=0.3)

        # Define fully-connected layers
        self.fc_dim = fc_dim
        if self.fc_dim > 0:
            self.fc = nn.Linear(channel_dims[-1], fc_dim)
            self.fc_out = nn.Linear(fc_dim, num_classes)
        else:
            self.fc_out = nn.Linear(channel_dims[-1], num_classes)

    def forward(self, data):
        x = data.x

        # Compute graph convolutional part
        if self.net_type != 'gmmcn':
            for gcn_layer in self.gcn:
                x = F.relu(gcn_layer(x, data.edge_index))
        else:
            for gcn_layer in self.gcn:
                x = F.relu(gcn_layer(x.float(), data.edge_index.long(), data.pseudo.float()))
        
        # Apply global sum pooling and dropout
        x = global_add_pool(x, data.batch)
        x = self.drop(x)
        embedding = x

        # Compute fully-connected part
        if self.fc_dim > 0:
            x = F.relu(self.fc(x))

        output = self.fc_out(x)   # sigmoid in loss function

        return embedding, output



class CNN1D_dilated(nn.Module):
    def __init__(self, input_dim=1024, num_filters=[64, 512], filter_sizes=[5, 5], fc_dim=256, num_classes=256):
        super(CNN1D_dilated, self).__init__()

        # Define 1D convolutional layers
        cnn_dims = [input_dim] + num_filters
        cnn_layers = [nn.Conv1d(cnn_dims[i-1], cnn_dims[i], kernel_size=filter_sizes[i-1], padding=int(filter_sizes[i-1]/2)*2, dilation=2)
                      for i in range(1, len(cnn_dims))]
        self.cnn = nn.ModuleList(cnn_layers)

        # Define global max pooling
        self.globalpool = nn.AdaptiveMaxPool1d(1)
        
        # Define dropout
        self.drop = nn.Dropout(p=0.3)

        # Define fully-connected layers
        self.fc_dim = fc_dim
        if self.fc_dim > 0:
            self.fc = nn.Linear(num_filters[-1], fc_dim)
            self.fc_out = nn.Linear(fc_dim, num_classes)
        else:
            self.fc_out = nn.Linear(num_filters[-1], num_classes)

    def forward(self, data):
        x = data.x

        # Compute 1D convolutional part
        for cnn_layer in self.cnn:
            x = F.relu(torch.mul(cnn_layer(x), data.mask))   # apply mask

        # Apply global max pooling and dropout
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        embedding = x

        # Compute fully-connected part
        if self.fc_dim > 0:
            x = F.relu(self.fc(x))
        
        output = self.fc_out(x)   # sigmoid in loss function

        return embedding, output



class CNN2D(nn.Module):
    def __init__(self, input_dim=1, num_filters=[64, 512], filter_sizes=[5, 5], fc_dim=512, num_classes=256):
        super(CNN2D, self).__init__()

        # Define 2D convolutional layers
        cnn_dims = [input_dim] + num_filters
        cnn_layers = [nn.Conv2d(cnn_dims[i-1], cnn_dims[i], kernel_size=filter_sizes[i-1], padding=int(filter_sizes[i-1]/2))
                        for i in range(1, len(cnn_dims))]
        self.cnn = nn.ModuleList(cnn_layers)

        # Define global max pooling
        self.globalpool = nn.AdaptiveMaxPool2d((1,1))

        # Define dropout
        self.drop = nn.Dropout(p=0.3)

        # Define fully-connected layers
        self.fc_dim = fc_dim
        if self.fc_dim > 0:
            self.fc = nn.Linear(num_filters[-1], fc_dim)
            self.fc_out = nn.Linear(fc_dim, num_classes)
        else:
            self.fc_out = nn.Linear(num_filters[-1], num_classes)

    def forward(self, data):
        x = data.x

        # Compute 2D convolutional part
        for cnn_layer in self.cnn:
            x = F.relu(torch.mul(cnn_layer(x), data.mask))   # apply mask

        # Apply global max pooling and dropout
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        embedding = x

        # Compute fully-connected part
        if self.fc_dim > 0:
            x = F.relu(self.fc(x))
        
        output = self.fc_out(x)   # sigmoid in loss function

        return embedding, output



class CNN1D2D(nn.Module):
    def __init__(self, input_dim_1d=1024, num_filters_1d=[64, 512], filter_sizes_1d=[5, 5],
                 input_dim_2d=1, num_filters_2d=[64, 512], filter_sizes_2d=[5, 5],
                 fc_dim=512, num_classes=256):
        super(CNN1D2D, self).__init__()

        # Define 1D convolutional layers
        cnn1d_dims = [input_dim_1d] + num_filters_1d
        cnn1d_layers = [nn.Conv1d(cnn1d_dims[i-1], cnn1d_dims[i], kernel_size=filter_sizes_1d[i-1], padding=int(filter_sizes_1d[i-1]/2)*2, dilation=2)
                        for i in range(1, len(cnn1d_dims))]
        self.cnn1d = nn.ModuleList(cnn1d_layers)

        # Define 1D global max pooling
        self.globalpool1d = nn.AdaptiveMaxPool1d(1)

        # Define 2D convolutional layers
        cnn2d_dims = [input_dim_2d] + num_filters_2d
        cnn2d_layers = [nn.Conv2d(cnn2d_dims[i-1], cnn2d_dims[i], kernel_size=filter_sizes_2d[i-1], padding=int(filter_sizes_2d[i-1]/2))
                        for i in range(1, len(cnn2d_dims))]
        self.cnn2d = nn.ModuleList(cnn2d_layers)

        # Define 2D global max pooling
        self.globalpool2d = nn.AdaptiveMaxPool2d((1,1))

        # Define dropout
        self.drop = nn.Dropout(p=0.3)

        # Define fully-connected layers
        self.fc_dim = fc_dim
        if self.fc_dim > 0:
            self.fc = nn.Linear(num_filters_1d[-1] + num_filters_2d[-1], fc_dim)
            self.fc_out = nn.Linear(fc_dim, num_classes)
        else:
            self.fc_out = nn.Linear(num_filters_1d[-1] + num_filters_2d[-1], num_classes)

    def forward(self, data):
        x1d = data.emb
        x2d = data.x
        
        mask1d = data.mask1d
        mask2d = data.mask

        # Compute 1D convolutional part
        for cnn1d_layer in self.cnn1d:
            x1d = F.relu(torch.mul(cnn1d_layer(x1d), mask1d))   # apply mask
        # Apply global max pooling and dropout
        x1d = self.globalpool1d(x1d)
        x1d = torch.flatten(x1d, 1)
        
        # Compute 2D convolutional part
        for cnn2d_layer in self.cnn2d:
            x2d = F.relu(torch.mul(cnn2d_layer(x2d), mask2d))   # apply mask
        # Apply 2D global max pooling
        x2d = self.globalpool2d(x2d)
        x2d = torch.flatten(x2d, 1)
        
        # Concatenate 1D-CNN and 2D-CNN outputs
        x = torch.cat((x1d, x2d), axis=1)
        embedding = x

        # Compute fully-connected part and apply dropout
        if self.fc_dim > 0:
            x = F.relu(self.fc(x))
        x = self.drop(x)
                
        output = self.fc_out(x)   # sigmoid in loss function

        return embedding, output



class Perceptron(nn.Module):
    def __init__(self, input_dim=1024, fc_dim=512, num_classes=256):
        super(Perceptron, self).__init__()

         # Define fully-connected layers and dropout
        self.layer1 = nn.Linear(input_dim, fc_dim)
        self.drop = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(fc_dim, num_classes)

    def forward(self, data):
        x = data.x

        # Compute fully-connected part and apply dropout
        x = F.relu(self.layer1(x))
        x = self.drop(x)
        embedding = x
        output = self.layer2(x)   # sigmoid in loss function

        return embedding, output