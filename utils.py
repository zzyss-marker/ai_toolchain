import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import base64
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        layers = []
        in_channels = 1  # MNIST数据集为灰度图，通道数为1

        for layer_cfg in config:
            layer_type = layer_cfg['type']
            params = layer_cfg['params']
            if layer_type == 'Conv2D':
                # 设置默认参数
                out_channels = params.get('out_channels', 32)
                kernel_size = params.get('kernel_size', 3)
                padding = params.get('padding', 0)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
                in_channels = out_channels
            elif layer_type == 'MaxPool2D':
                kernel_size = params.get('kernel_size', 2)
                stride = params.get('stride', 2)
                layers.append(nn.MaxPool2d(kernel_size, stride=stride))
            elif layer_type == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_type == 'Flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'Linear':
                out_features = params.get('out_features', 10)
                # in_features需要根据前一层计算
                in_features = self._infer_in_features(layers)
                layers.append(nn.Linear(in_features, out_features))
            # 可以添加更多的层类型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _infer_in_features(self, layers):
        # 使用虚拟输入推断特征数量
        x = torch.zeros(1, 1, 28, 28)
        for layer in layers:
            x = layer(x)
        return x.view(1, -1).size(1)


def train_model(model, train_loader, criterion, optimizer, epochs, update_progress_fn):
    # 添加数据增强
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    model.train()
    training_loss = []
    training_accuracy = []
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 应用数据增强
            data = torch.stack([transform(d) for d in data])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            current_step += 1
            progress = int(100 * current_step / total_steps)
            update_progress_fn(progress)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        training_loss.append(avg_loss)
        training_accuracy.append(accuracy)
    
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
    
    accuracy = 100. * correct / total
    return accuracy

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
