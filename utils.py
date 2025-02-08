import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import base64
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, config, input_shape=(1, 28, 28), num_classes=10):
        super(Net, self).__init__()
        self.input_shape = input_shape
        layers = []
        in_channels = input_shape[0]  # 根据输入形状确定通道数

        for layer_cfg in config:
            layer_type = layer_cfg['type']
            params = layer_cfg['params']
            if layer_type == 'Conv2D':
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
            elif layer_type == 'Dropout':
                p = params.get('p', 0.5)
                layers.append(nn.Dropout(p))
            elif layer_type == 'Flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'Linear':
                out_features = params.get('out_features', num_classes)
                in_features = self._infer_in_features(layers)
                layers.append(nn.Linear(in_features, out_features))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _infer_in_features(self, layers):
        # 创建一个示例输入，使用模型的输入形状
        x = torch.zeros(1, *self.input_shape)
        # 跟踪每一层的输出形状
        print(f"Initial shape: {x.shape}")
        for layer in layers:
            x = layer(x)
            print(f"After {layer.__class__.__name__}: {x.shape}")
        # 如果是展平的张量，直接使用 view 后的大小
        if len(x.shape) == 2:
            return x.size(1)
        # 如果还是多维张量，先展平再获取特征数
        return x.view(1, -1).size(1)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, update_progress_fn, patience=5):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    total_steps = epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            current_step += 1
            progress = int(100 * current_step / total_steps)
            update_progress_fn(progress)

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)  # 恢复最佳模型
                break
        
        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return metrics

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
