import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import base64

class Net(nn.Module):
    def __init__(self, conv_layers=2, conv_filters=32, kernel_size=3, pool_size=2, dense_units=64):
        super(Net, self).__init__()
        layers = []
        in_channels = 1
        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, conv_filters, kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(pool_size))
            in_channels = conv_filters
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(conv_filters * ((28 - (kernel_size - 1) * conv_layers) // (pool_size ** conv_layers)) ** 2, dense_units)
        self.fc2 = nn.Linear(dense_units, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        return x.size(1) * x.size(2) * x.size(3)

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    training_loss = []
    training_accuracy = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        training_loss.append(epoch_loss / len(train_loader))
        training_accuracy.append(100. * correct / total)
    return training_loss, training_accuracy

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

def plot_training_curve(data, label):
    plt.figure()
    plt.plot(data)
    plt.title(f'Training {label}')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()
    return img_base64
