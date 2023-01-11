import torch
import torch.nn as nn
import torch.optim as optimize


class X_train_gray:
    pass


class TheModelClass(nn.Module):

    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv = nn.Conv2d(4, 7, 6)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv1 = nn.Conv2d(7, 17, 6)
        self.fc1 = nn.Linear(17 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv(X)))
        X = self.pool(F.relu(self.conv1(X)))
        X = X.view(-1, 17 * 5 * 5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X_train_gray
