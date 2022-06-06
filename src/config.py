import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
SPARSITY = 0.6
OUTPUT_DIM = 10
N_ITER_TRAIN = 500
N_MASKS = 5000
MODEL_PATH = "../model/model_mnist.pt"
PRUNED_MODEL_PATH = "../model/pruned_model_mnist.pt"
alpha = 0.1
n_bayesian_iter = 40
n_bayesian_init_points = 10