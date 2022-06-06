import config
import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets


seed = 142
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import torch
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(),
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.ToTensor(),
    download = True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

config.HIDDEN_DIM = 128
config.SPARSITY = 0.9
config.OUTPUT_DIM = 10
config.N_ITER_TRAIN = 500
config.N_MASKS = 5000

X_train = train_data.data.reshape(-1, 28*28).float().to(device)
y_train = train_data.targets.to(device)
X_test = test_data.data.reshape(-1, 28*28).float().to(device)
y_test = test_data.targets.to(device)
INP_DIM = X_train.shape[1]


class LogisticRegressionModel(nn.Module):
    def __init__(self, INP_DIM=INP_DIM, HIDDEN_DIM=config.HIDDEN_DIM, OUTPUT_DIM=config.OUTPUT_DIM):
        torch.manual_seed(seed) #return the same model always
        super(LogisticRegressionModel, self).__init__()
        self.HIDDEN_DIM = config.HIDDEN_DIM
        self.INP_DIM = INP_DIM
        self.OUTPUT_DIM = config.OUTPUT_DIM 
        self.linear1 = nn.Linear(INP_DIM, config.HIDDEN_DIM)
        self.linear2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM//2)
        self.linear3 = nn.Linear(config.HIDDEN_DIM//2, config.HIDDEN_DIM//4)
        self.linear4 = nn.Linear(config.HIDDEN_DIM//4, config.OUTPUT_DIM)     

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=1)
        return x

    def get_layer_sizes(self):
      return [self.INP_DIM, self.HIDDEN_DIM, self.HIDDEN_DIM//2, self.HIDDEN_DIM//4, self.OUTPUT_DIM]

def _train(model, epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
  criterion = nn.CrossEntropyLoss()
  loss_list  = np.zeros((epochs,))
  accuracy_list = np.zeros((epochs,))

  for epoch in range(epochs):
      y_pred = model(X_train)
      loss = criterion(y_pred, y_train)
      loss_list[epoch] = loss.item()
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      with torch.no_grad():
          y_pred = model(X_test)
          correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
          accuracy_list[epoch] = correct.mean()

  return model, accuracy_list, loss_list


def plot(accuracy_list, loss_list):
  fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

  ax1.plot(accuracy_list)
  ax1.set_ylabel("validation accuracy")
  ax2.plot(loss_list)

  ax2.set_ylabel("validation loss")
  ax2.set_xlabel("epochs");     


def _model_performance(model, N_ITERS=50, train=False):
  if train:
    model, _, _ = _train(model, N_ITERS)
  with torch.no_grad():
    y_pred = model(X_test)
    correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
    accuracy = correct.mean()
  return accuracy 
