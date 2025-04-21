import time
import sys
import os
import argparse
import io
from datetime import datetime
import numpy as np
import awkward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, DynamicEdgeConv, global_mean_pool

# Defintion of the GNN model
# Use the DynamicEdgeConv layer from the pytorch geometric package like this:
# MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
# The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
# because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
# The output dimension of the MLP is the new feauture dimension of this graph layer.
from torch_geometric.nn import DynamicEdgeConv
class GNNEncoder(nn.Module):
    def __init__(self):
        super(GNNEncoder, self).__init__()
        
        k=3
        
        self.MLP1 = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.ReLU()
        )
        layer1 = DynamicEdgeConv(
                    self.MLP1,
                    aggr='mean', k=k,  # k is the number of nearest neighbors to consider
                )
        
        self.MLP2 = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU()
        )
        
        layer2 = DynamicEdgeConv(
                    self.MLP2,
                    aggr='mean', k=k,  # k is the number of nearest neighbors to consider
                )
        
        
        self.MLP3 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        
        layer3 = DynamicEdgeConv(
                    self.MLP3,
                    aggr='mean', k=k,  # k is the number of nearest neighbors to consider
                )
        self.layer_list = [layer1, layer2, layer3]
        
        self.final_mlp = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,2),
        )
        
    def forward(self, data):
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layer_list:
            x = layer(x, batch)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_feature_dimension)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node feauters into single prediction, we recommend to use global pooling
        x = global_mean_pool(x, batch) # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # either your the last graph feature dimension is already the output dimension you want to predict
        # or you need to add a final MLP layer to map the output dimension to the number of labels you want to predict
        x = self.final_mlp(x)

        return x