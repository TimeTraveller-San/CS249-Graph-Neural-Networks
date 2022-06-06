# Bayes Optimal Neural Network Pruning using Relational Graph Properties

[Colab Demo](https://colab.research.google.com/drive/11OTVuHP2gDzyToAz-PJItv27XRmxSUXi?usp=sharing) | [Presentation Colab](https://github.com/yichousun/Spring2022_CS249_GNN/blob/main/Course_Project/Group15_NN_Pruning_via_Relational_Graph/Vector_Bayesian_Opimization.ipynb) | [Presentation Slides](https://docs.google.com/presentation/d/1gqfWB3UVYjL8rVKUo8p64ba-yEdD2MGkyxg_0BZdEXo/edit)

| Name       | UID |
|---------------|----------|
| Andrei Rekesh |   105343158  |
| Sahil Bansal | 905525442     |
| Vaibhav Kumar | 305710616 |
| Vivek Arora   | 505526269 |

## Abstract
Graph representations of neural networks are commonly used to assist in training-time computations like backpropagation and weight updates. However, recent works have proposed a novel, relational graph representation allowing network depth to correspond to rounds of message passing. The correlation between neural network performance and two key properties of these relational graphs, average shortest path length (ASPL) and clustering coefficient, has been previously investigated by randomly generating graphs spanning a wide range of both properties and exhaustively evaluating performance. Here, we extend this analysis to five properties: average degree centrality, eigenvector centrality, clustering coefficient, edge connectivity, and number of cycles. Rather than take a large number of samples from the five-dimensional graph property space and evaluate performance directly, we perform Bayesian optimization over these properties using a small randomly sampled database of graphs (n = 1000) to determine the best values for each. We find that these properties are important to finding high-performing candidate neural network pruning masks, that such masks can be identified efficiently, and that the trade-off between pruning and accuracy can be learned adaptively.

## Setup
- [Python 3.9](https://www.python.org/)
- `requirements.txt` contains all the required python libraries.  
- `pip install -r requirements.txt`

## Files
- `src/run.py`: Run Bayesian Optimization Mask Learning
- `src/utils.py`: Utils for fetching masks
- `src/graph.py`: Compute grpah properties to create graph db
- `src/config.py`: Config for training base model and bayesian optimization
- `src/train`: Directory for training base models
- `src/train/mnist.py`: Directory for training MNIST models

# How to run:
1. Try running the colab demo shared above, it has all the code and environment compiled in a single jupyter NB
2. To run baselines, run the colab notebook. This repo only contains our implementation of NN pruning.
5. Run `python3.9 run.py`