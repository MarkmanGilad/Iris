import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load Iris dataset
iris = datasets.load_iris()
data, targets = iris.data, iris.target

print(iris.DESCR)

print ("target_names",iris.target_names)
print ("X, y",data.shape, targets.shape)
print(data[0:5],'\n', targets)

# change labels: stosa = 1 else = 0
targets[targets == 0] = -1
targets[targets != -1] = 0
targets[targets == -1] = 1
print(targets)
