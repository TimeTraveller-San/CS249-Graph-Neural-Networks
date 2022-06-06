from bayes_opt import BayesianOptimization
from utils import *
import graph
from train.mnist import *
import train
from graph import *
import config

import numpy as np
import sklearn

import torch
import tqdm


seed = 142


mask_vecs = []
mask_db = []

def generateDB(model):
    global mask_vecs
    global mask_db
    MASK_DIM = min(model.get_layer_sizes())
    mask_db = []
    for SPARSITY in [config.SPARSITY]: #for loop in case general db needs to be created
        for _ in tqdm.tqdm(range(config.N_MASKS)):
            mask = random_mask(MASK_DIM, config.SPARSITY)
            G = mask_to_graph(mask)  
            try:
                ps = graph.compute_properties(G)
                mask_db.append([mask, ps])
            except:
                continue

    assert len(mask_db[0][1])==len(properties)
    mask_vecs = np.array([item[1] for item in mask_db])
    mins = mask_vecs.min(0)
    maxs = mask_vecs.max(0)
    ranges = {
        list(properties.keys())[i]:(mins[i], maxs[i]) for i in range(len(properties))
    }
    for p in ranges:
        print(f"{p}: {ranges[p][0]}-{ranges[p][1]}")

    M = mask_vecs.ptp(0)
    D = mask_vecs.min(0)
    mask_vecs = (mask_vecs - D) / M
    mask_vecs = sklearn.preprocessing.normalize(mask_vecs, norm="l2")

    #Store the normalized range isntead --> Better bayesian opti
    mins = mask_vecs.min(0)
    maxs = mask_vecs.max(0)
    ranges = {
        list(properties.keys())[i]:(mins[i], maxs[i]) for i in range(len(properties))
    }
    print(mask_vecs.shape)  
    return ranges      


def train_baseline():
    model = LogisticRegressionModel().to(config.device)
    gmask = get_mask(model.get_layer_sizes(), config.SPARSITY)
    apply_mask(model, gmask)
    sparsity = get_model_sparsity(model)
    model = LogisticRegressionModel().to(config.device)
    train.mnist._train(model, config.N_ITER_TRAIN)
    acc = train.mnist._model_performance(model)
    print(f"Baseline model accuracy: {acc}")
    print(f"Baseline model sparsity: {sparsity}")
    torch.save(model.state_dict(), config.MODEL_PATH)
    return model    



def f(**kwargs):
  p = np.array(list(kwargs.values()))

  #Load pretrained model
  model = LogisticRegressionModel().to(config.device)
  model.load_state_dict(torch.load(config.MODEL_PATH))

  #Fetch mask
  mask, _ = fetch_mask(p, mask_vecs, mask_db)

  #Apply mask
  apply_mask(model, mask)

  #Get performance
  performance = train.mnist._model_performance(model, N_ITERS=config.N_ITER_TRAIN, train=False)

  #Get Sparsity
  sparsity = get_model_sparsity(model)

  return (1-config.alpha)*performance + config.alpha*sparsity


def f2(**kwargs):
  p = np.array(list(kwargs.values()))

  #Load pretrained model
  model = LogisticRegressionModel().to(config.device)
  model.load_state_dict(torch.load(config.MODEL_PATH))

  #Fetch mask
  mask, _ = fetch_mask(p, mask_vecs, mask_db)

  #Apply mask
  apply_mask(model, mask)

  #Get performance
  return train.mnist._model_performance(model, N_ITERS=config.N_ITER_TRAIN, train=True)


def main():
    print("Training baseline model without pruning...")
    model = train_baseline()
    print("Generating DB...")
    ranges = generateDB(model)
    print("Running Bayesian Optimization")
    optimizer = BayesianOptimization(
        f2, #f2 is default for a sparsity level defined in config, can change to f to learn sparisty according to alpha
        ranges,
        random_state=42
    )
    optimizer.maximize(init_points=config.n_bayesian_init_points, n_iter=config.n_bayesian_iter)   
    winner_ticket = optimizer.max
    winner_mask = fetch_mask(list(winner_ticket['params'].values()), mask_vecs, mask_db)[0]

    #Load pretrained model
    model = LogisticRegressionModel().to(config.device)
    model.load_state_dict(torch.load(config.MODEL_PATH))

    #Apply Winning Mask
    apply_mask(model, winner_mask)
    performance = train.mnist._model_performance(model, N_ITERS=config.N_ITER_TRAIN, train=True)
    sparsity = get_model_sparsity(model)
    print(f"Performance After Pruning: {performance}")
    print(f"Sparsity After Pruning: {sparsity}")
    torch.save(model.state_dict(), config.PRUNED_MODEL_PATH)
    print(f"Saved pruned model at: {config.PRUNED_MODEL_PATH}")


main()