import config
from train.mnist import *
import numpy as np
import networkx as nx
import itertools
import copy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec


import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


from torchvision import datasets
from torchvision.transforms import ToTensor


import torch.nn.utils.prune as prune
import random

seed = 142


def get_layer_sparsity(layer):
  with torch.no_grad():
    s = layer.weight.data.nonzero().size(0)/layer.weight.data.numel()
  return 1-s
  
def compute_size(channel, group):
  divide = channel // group
  remain = channel % group
  out = np.zeros(group, dtype=int)
  out[:remain] = divide + 1
  out[remain:] = divide
  return out
    
def graph_to_mask(gmask, in_channels, out_channels, group_num):
  repeat_in = compute_size(in_channels, group_num)
  repeat_out = compute_size(out_channels, group_num)
  mask_NN = np.repeat(gmask.detach().cpu(), repeat_out, axis=0)
  mask_NN = np.repeat(mask_NN, repeat_in, axis=1)
  return mask_NN.to(config.device)

def random_mask(hd, sparsity):
  mask = np.random.uniform(size=(hd, hd))
  mask = np.triu(mask, k=0)
  mask = (mask > sparsity).astype(int)
  mask = np.maximum(mask, mask.T)
  mask = torch.from_numpy(mask)
  return mask.to(config.device)

def get_mask(layer_sizes, sparsity):
  num_node = min(layer_sizes)
  return random_mask(num_node, sparsity)

def get_model_sparsity(model):
  with torch.no_grad():
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
  return 1-(nnzs/prunables)

def apply_mask(model, gmask):
  layer_sizes = model.get_layer_sizes()
  num_nodes = min(layer_sizes)
  for m in model.modules():
    if _is_prunable_module(m):
      d_in, d_out = m.weight.data.shape
      mask = graph_to_mask(gmask, d_out, d_in, num_nodes).to(config.device)
      prune.custom_from_mask(m, name='weight', mask=mask)
  return model.to(config.device)

def model_performance(mask, N_ITERS=50):
  model = LogisticRegressionModel(INP_DIM, HIDDEN_DIM, OUTPUT_DIM).to(config.device)
  mask = mask.to(config.device)
  model = apply_mask(model, mask)
  # s = (get_layer_sparsity(model.linear2) + get_layer_sparsity(model.linear3))/2
  # print(f"Sparsity: {s}")
  return _model_performance(model, N_ITERS), model 

def model_performance_no_train(mask, model):
  mask = mask.to(config.device)
  model = apply_mask(model, mask)
  return _model_performance(model, train=False), model   

def mask_to_graph(mask):
  mask = mask.cpu().numpy()
  return nx.from_numpy_array(mask)

def _is_prunable_module(m):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d))
    
def get_model_sparsity(model):
  with torch.no_grad():
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
  return 1-(nnzs/prunables)  


properties = {
    "avg_degree_centrality": lambda x: np.array(list(nx.degree_centrality(x).values())).mean(),
    "average_degree_connectivity": lambda x: np.array(list(nx.average_degree_connectivity(x).values())).mean(),
    "eigenvector_centrality": lambda x: np.array(list(nx.eigenvector_centrality(x).values())).mean(),
    "clustering_coefficient": lambda x: nx.average_clustering(x),
    "average_shortest_path_length": lambda x: nx.average_shortest_path_length(x),
    "s_metric": lambda x: nx.s_metric(x, False),
    "wiener_index": lambda x: nx.wiener_index(x),
    "edge_connectivity": lambda x: nx.edge_connectivity(x),
    "n_cycles": lambda x: len(nx.find_cycle(x))
}

drops = [
         "average_degree_connectivity",
         "s_metric",
         "average_shortest_path_length",
         "wiener_index",
]

properties = {k:properties[k] for k in properties if k not in drops}

def compute_properties(G):
  return [p(G) for p in list(properties.values())]

properties    