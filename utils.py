import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import base64
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, model_config, input_shape, num_classes):
        super(Net, self).__init__()
        # 使用 Sequential 来构建模型
        layers = []
        self.layer_names = []
        self.intermediate_outputs = {}
        
        current_shape = input_shape
        for layer_config in model_config:
            layer_type = layer_config['type']
            layer_params = layer_config['params']
            
            if layer_type == 'Conv2D':
                layer = nn.Conv2d(current_shape[0], layer_params['out_channels'],
                                kernel_size=layer_params.get('kernel_size', 3),
                                padding=layer_params.get('padding', 0),
                                stride=layer_params.get('stride', 1))
                current_shape = (layer_params['out_channels'],
                               (current_shape[1] + 2*layer_params.get('padding', 0) - layer_params.get('kernel_size', 3))//layer_params.get('stride', 1) + 1,
                               (current_shape[2] + 2*layer_params.get('padding', 0) - layer_params.get('kernel_size', 3))//layer_params.get('stride', 1) + 1)
            elif layer_type == 'MaxPool2D':
                kernel_size = layer_params.get('kernel_size', 2)
                stride = layer_params.get('stride', kernel_size)
                layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                current_shape = (current_shape[0],
                               (current_shape[1] - kernel_size)//stride + 1,
                               (current_shape[2] - kernel_size)//stride + 1)
            elif layer_type == 'Flatten':
                layer = nn.Flatten()
                current_shape = (current_shape[0] * current_shape[1] * current_shape[2],)
            elif layer_type == 'Linear':
                in_features = current_shape[0] if isinstance(current_shape, tuple) else current_shape
                layer = nn.Linear(in_features, layer_params['out_features'])
                current_shape = layer_params['out_features']
            elif layer_type == 'ReLU':
                layer = nn.ReLU()
            elif layer_type == 'Dropout':
                layer = nn.Dropout(layer_params.get('p', 0.5))
            else:
                raise ValueError(f'Unknown layer type: {layer_type}')
            
            layers.append(layer)
            self.layer_names.append(layer_type)

        # 创建顺序模型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        print("\n" + "="*50)
        print("推理过程详解")
        print("="*50)
        print(f"\n输入数据形状: {x.shape}")
        print(f"输入数据范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
        
        self.intermediate_outputs = {'input': x.detach().cpu()}
        x_current = x
        for i, (name, layer) in enumerate(zip(self.layer_names, self.model)):
            print("\n" + "-"*30)
            print(f"第 {i+1} 层: {name}")
            print("-"*30)
            
            # 显示层的参数
            if isinstance(layer, nn.Conv2d):
                print(f"类型: 卷积层")
                print(f"输入通道数: {layer.in_channels}")
                print(f"输出通道数: {layer.out_channels}")
                print(f"卷积核大小: {layer.kernel_size}")
                print(f"步长: {layer.stride}")
                print(f"填充: {layer.padding}")
                print("\n计算公式:")
                print("输出大小 = (输入大小 - 核大小 + 2×填充) / 步长 + 1")
            elif isinstance(layer, nn.MaxPool2d):
                print(f"类型: 最大池化层")
                print(f"池化窗口大小: {layer.kernel_size}")
                print(f"步长: {layer.stride}")
                print("\n操作: 在每个窗口中取最大值")
            elif isinstance(layer, nn.ReLU):
                print(f"类型: ReLU激活函数")
                print("\n计算公式: f(x) = max(0, x)")
            elif isinstance(layer, nn.Flatten):
                print(f"类型: 展平层")
                print("\n操作: 将多维特征图展平为一维向量")
            elif isinstance(layer, nn.Linear):
                print(f"类型: 全连接层")
                print(f"输入维度: {layer.in_features}")
                print(f"输出维度: {layer.out_features}")
                print("\n计算公式: y = Wx + b")
            
            x_current = layer(x_current)
            print(f"\n输出形状: {x_current.shape}")
            print(f"输出数据范围: [{x_current.min().item():.3f}, {x_current.max().item():.3f}]")
            
            # 对于卷积层和池化层，显示特征图的统计信息
            if len(x_current.shape) == 4:  # 是特征图
                avg_activation = x_current.mean().item()
                std_activation = x_current.std().item()
                print(f"平均激活值: {avg_activation:.3f}")
                print(f"激活值标准差: {std_activation:.3f}")
            
            self.intermediate_outputs[f'{name}_{i}'] = x_current.detach().cpu()
        
        return x_current

    def get_intermediate_outputs(self):
        print("Getting intermediate outputs")
        print("Available outputs:", list(self.intermediate_outputs.keys()))
        return self.intermediate_outputs

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
