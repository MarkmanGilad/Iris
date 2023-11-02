import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load Iris dataset
iris = datasets.load_iris()
data, targets = iris.data, iris.target

print(iris.DESCR)

print ("target_names",iris.target_names)
print ("data, labels",data.shape, targets.shape)
print(data[0:5],'\n', targets)

# change labels: stosa = 1 else = 0
# targets[targets == 0] = -1
# targets[targets != -1] = 0
# targets[targets == -1] = 1
# print(targets)

#prepare data
n_samples, n_features = data.shape
data_train_np, data_test_np, targets_train_np, targets_test_np = train_test_split (data,targets, test_size=0.2, random_state = 2, shuffle=True)

data_train = torch.from_numpy(data_train_np.astype(np.float32))
targets_train = torch.from_numpy(targets_train_np.astype(np.int64))
# targets_train = targets_train.view(-1,1)

data_test = torch.from_numpy(data_test_np.astype(np.float32))
targets_test = torch.from_numpy(targets_test_np.astype(np.int64))
# targets_test = targets_test.view(-1,1)
print(data_train.shape, targets_train.shape)

#Normalized

#parametes
learning_rate = 0.1
epochs = 500
losses = []

# design model
class Iris_NN (nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 3)

    def forward (self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

Model = Iris_NN()

#construct loss and optimizer
Loss = nn.CrossEntropyLoss()

# init optimizer
optim = torch.optim.Adam(Model.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
    # forward
    targets_predict = Model(data_train)

    # backward
    optim.zero_grad()
    loss = Loss(targets_predict, targets_train)
    loss.backward()

    if epoch % 10 == 0:
        print(f"epoch= {epoch} loss={loss.item():.4f} ")

    # update wights
    optim.step()
    losses.append(loss.item())

# print results
plt.plot(losses, 'o')
plt.show() 

#test
print("\n\n")
with torch.no_grad():
    targets_predicted = Model(data_test)
    _, target_predicred_arg = torch.max(targets_predicted,1)
    correct = target_predicred_arg.eq(targets_test).sum() / float(targets_test.shape[0])
    print (f'accuracy = {correct:.4f}', 'sampels', targets_test.shape[0])

for i in range(len(targets_test)):
    print('targets test=',targets_test[i].item(),'\ttarget predicted=', target_predicred_arg[i].item(), '\t',targets_test[i].item() == target_predicred_arg[i].item() )
    
  