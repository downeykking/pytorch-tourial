import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
num_classes = 2
batch_size = 1
learning_rate = 0.01
input_size = 2


# Dataset
class mydata(Dataset):
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter=',')

    def __getitem__(self, index):
        self.x = self.data[:, 0:-1]
        self.label = self.data[:, [-1]]
        return self.x[index], self.label[index]

    def __len__(self):
        return len(self.data)


train_dataset = mydata("ex2data1.txt")

# Train loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)


# Network
class LR(nn.Module):
    def __init__(self, input_size=2):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        # out = out.squeeze(-1)
        return out


model = LR(input_size)
model = model.to(device)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for x, labels in train_loader:
        x = x.float()
        x = x.to(device)
        labels = labels.float()
        labels = labels.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch+1) % total_step == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                   loss.item()))
