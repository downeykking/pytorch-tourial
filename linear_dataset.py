import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 500
num_classes = 2
batch_size = 100
learning_rate = 0.01
input_size = 30

# Data

breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Dataset
class train_data(Dataset):
    def __init__(self):
        scaler = sklearn.preprocessing.StandardScaler()
        self.x_train = scaler.fit_transform(x_train)

    def __getitem__(self, index):
        self.x = torch.from_numpy(self.x_train.astype(np.float32))
        self.label = torch.from_numpy(y_train.astype(np.float32))
        return self.x[index], self.label[index]

    def __len__(self):
        return len(self.x_train)


class test_data(Dataset):
    def __init__(self):
        scaler = sklearn.preprocessing.StandardScaler()
        self.x_test = scaler.fit_transform(x_test)

    def __getitem__(self, index):
        self.x = torch.from_numpy(self.x_test.astype(np.float32))
        self.label = torch.from_numpy(y_test.astype(np.float32))
        return self.x[index], self.label[index]

    def __len__(self):
        return len(self.x_test)


train_dataset = train_data()
test_dataset = test_data()

# Train loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# Network
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(30, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.squeeze(-1)
        return out


model = LR()
model = model.to(device)

# # Loss and optimizer
# # nn.CrossEntropyLoss() computes softmax internally
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # Train the model
# total_step = len(train_loader)
scaler = sklearn.preprocessing.StandardScaler()
x_test = scaler.fit_transform(x_test)
x_test = torch.from_numpy(x_test.astype(np.float32))
x_test = x_test.to(device)

for epoch in range(num_epochs):
    model.train()
    t_loss = 0
    for x, labels in train_loader:
        x = x.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, labels)
        t_loss += loss.detach().cpu().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            y_p = model(x_test)
            y_p = y_p.round()
            acc = (y_p.cpu().eq(torch.from_numpy(y_test)).sum()) / float(
                y_test.shape[0])
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}'.format(
                epoch + 1, num_epochs, t_loss, acc))
    t_loss = 0
