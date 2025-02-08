import os
# 设置环境变量，必须在导入其他库之前
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from flask import Flask, render_template, request, jsonify, url_for
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import io
import base64
from utils import Net, train_model, test_model, plot_training_curve
from PIL import Image
from PIL import ImageOps
import os
import json
import threading
from torch.utils.data import random_split
import torch.nn.functional as F

app = Flask(__name__)

# 全局变量存储训练曲线数据
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

# 全局变量存储训练进度
training_progress = 0
progress_lock = threading.Lock()

# 全局变量存储最后一次推理的数据
last_inference_data = None

AVAILABLE_DATASETS = {
    'MNIST': {
        'loader': datasets.MNIST,
        'input_shape': (1, 28, 28),
        'num_classes': 10
    },
    'CIFAR10': {
        'loader': datasets.CIFAR10,
        'input_shape': (3, 32, 32),
        'num_classes': 10
    },
    'FashionMNIST': {
        'loader': datasets.FashionMNIST,
        'input_shape': (1, 28, 28),
        'num_classes': 10
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_components', methods=['GET'])
def get_components():
    components = {
        'datasets': list(AVAILABLE_DATASETS.keys()),
        'modules': ['Conv2D', 'MaxPool2D', 'ReLU', 'Flatten', 'Linear', 'Dropout'],
        'saved_models': os.listdir('static/models') if os.path.exists('static/models') else [],
        'preprocessing': {
            'augmentation': ['RandomRotation', 'RandomHorizontalFlip', 'RandomCrop', 'ColorJitter'],
            'normalization': ['StandardNormalization', 'MinMaxNormalization']
        }
    }
    return jsonify(components)

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        print("Received training request:", data)
        
        model_config = data['model_config']
        training_params = data['training_params']
        dataset_name = data.get('dataset')
        preprocessing = data.get('preprocessing', {})

        # 过滤掉非模型层的配置
        model_config = [layer for layer in model_config 
                       if layer['type'] in ['Conv2D', 'MaxPool2D', 'ReLU', 'Flatten', 'Linear', 'Dropout']]

        print(f"Received dataset name: {dataset_name}")
        print(f"Available datasets: {list(AVAILABLE_DATASETS.keys())}")

        # 验证数据集是否存在
        if not dataset_name:
            print("Dataset name is missing")  # 添加日志
            return jsonify({'status': 'error', 'message': '未指定数据集'}), 400

        # 验证数据集名称是否有效
        if dataset_name not in AVAILABLE_DATASETS:
            print(f"Invalid dataset name: {dataset_name}")
            print(f"Valid dataset names: {list(AVAILABLE_DATASETS.keys())}")
            return jsonify({'status': 'error', 
                          'message': f'无效的数据集名称: {dataset_name}'}), 400

        if not model_config:
            return jsonify({'status': 'error', 'message': '请先添加模型层'}), 400

        # 检查模型结构是否合理
        has_flatten = False
        has_linear = False
        for layer in model_config:
            if layer['type'] == 'Flatten':
                has_flatten = True
            elif layer['type'] == 'Linear':
                has_linear = True
                if not has_flatten:
                    return jsonify({'status': 'error', 'message': 'Linear层前必须有Flatten层'}), 400

        if not has_linear:
            return jsonify({'status': 'error', 'message': '模型必须包含至少一个Linear层作为输出层'}), 400

        # 重置进度
        global training_progress
        training_progress = 0

        # 构建数据预处理流程
        transform_list = []
        
        # 添加数据增强
        if 'augmentation' in preprocessing:
            for aug in preprocessing['augmentation']:
                if aug == 'RandomRotation':
                    transform_list.append(transforms.RandomRotation(15))
                elif aug == 'RandomHorizontalFlip':
                    transform_list.append(transforms.RandomHorizontalFlip())
                elif aug == 'RandomCrop':
                    # 根据数据集调整裁剪大小
                    if dataset_name in ['MNIST', 'FashionMNIST']:
                        transform_list.append(transforms.RandomCrop(28, padding=4))
                    else:  # CIFAR10
                        transform_list.append(transforms.RandomCrop(32, padding=4))
                elif aug == 'ColorJitter':
                    # 只对 CIFAR10 使用颜色增强
                    if dataset_name == 'CIFAR10':
                        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

        # 添加基本转换
        # 确保调整到正确的大小
        if dataset_name in ['MNIST', 'FashionMNIST']:
            transform_list.append(transforms.Resize((28, 28)))
        else:  # CIFAR10
            transform_list.append(transforms.Resize((32, 32)))
        
        transform_list.append(transforms.ToTensor())
        
        # 添加标准化
        if 'normalization' in preprocessing:
            if preprocessing['normalization'] == 'StandardNormalization':
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            elif preprocessing['normalization'] == 'MinMaxNormalization':
                transform_list.append(transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())))

        transform = transforms.Compose(transform_list)

        # 加载数据集
        dataset_info = AVAILABLE_DATASETS[dataset_name]
        full_dataset = dataset_info['loader']('./data', train=True, download=True, transform=transform)
        test_dataset = dataset_info['loader']('./data', train=False, download=True, transform=transform)

        # 分割训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)

        # 构建模型
        model = Net(model_config, input_shape=dataset_info['input_shape'], num_classes=dataset_info['num_classes'])
        optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # 训练模型
        global training_loss, training_accuracy, validation_loss, validation_accuracy
        training_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            training_params['epochs'], update_progress,
            patience=training_params.get('early_stopping_patience', 5)
        )
        
        training_loss = training_metrics['train_loss']
        training_accuracy = training_metrics['train_acc']
        validation_loss = training_metrics['val_loss']
        validation_accuracy = training_metrics['val_acc']

        # 测试模型
        test_acc = test_model(model, test_loader)

        # 训练完成后保存模型
        if not os.path.exists('static/models'):
            os.makedirs('static/models')

        # 计算验证集准确率作为模型评分
        val_acc = validation_accuracy[-1]
        
        # 确保使用正确的数据集名称
        model_prefix = {
            'MNIST': 'MNIST',
            'CIFAR10': 'CIFAR10',
            'FashionMNIST': 'FashionMNIST'
        }[dataset_name]

        # 生成模型文件名
        model_name = f'{model_prefix}_acc{val_acc:.1f}_{len(os.listdir("static/models"))}.pth'
        
        # 保存完整模型状态
        save_dict = {
            'model_state': model.state_dict(),
            'model_config': model_config,
            'dataset': model_prefix,
            'preprocessing': preprocessing,
            'input_shape': dataset_info['input_shape'],
            'num_classes': dataset_info['num_classes']
        }
        
        print(f"Saving model for dataset: {model_prefix} as {model_name}")
        torch.save(save_dict, f'static/models/{model_name}')

        return jsonify({
            'status': 'success', 
            'model_name': model_name,
            'accuracy': test_acc,
            'redirect_url': url_for('training_results')
        })

    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/training_results', methods=['GET'])
def training_results():
    # 绘制训练曲线
    loss_img = plot_training_curve(training_loss, 'Loss')
    acc_img = plot_training_curve(training_accuracy, 'Accuracy')
    return render_template('train.html', loss_img=loss_img, acc_img=acc_img)

@app.route('/model_info/<model_name>')
def model_info(model_name):
    try:
        model_path = f'static/models/{model_name}'
        print(f"Loading model from: {model_path}")
        model_data = torch.load(model_path)
        
        # 获取模型结构字符串
        structure = []
        for layer in model_data['model_config']:
            params = layer['params']
            param_str = ', '.join(f'{k}={v}' for k, v in params.items())
            structure.append(f"{layer['type']}({param_str})")
        
        response = {
            'dataset': model_data['dataset'],
            'structure': '\n'.join(structure),
            'input_shape': model_data['input_shape']
        }
        print(f"Returning model info: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"Error in model_info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        try:
            file = request.files['image']
            model_name = request.form.get('model_name')
            
            # 保存当前推理的信息到全局变量
            global last_inference_data
            last_inference_data = {
                'input_image': None,
                'intermediate_outputs': None,
                'model_name': model_name
            }
            
            # 加载模型数据
            model_path = f'static/models/{model_name}'
            model_data = torch.load(model_path)
            dataset_type = model_data['dataset']
            
            print(f"数据集类型: {dataset_type}")
            print(f"模型输入形状: {model_data['input_shape']}")
            print(f"预处理配置: {model_data['preprocessing']}")

            # 过滤掉非模型层的配置
            valid_layer_types = ['Conv2D', 'MaxPool2D', 'ReLU', 'Flatten', 'Linear', 'Dropout']
            model_config = [layer for layer in model_data['model_config'] 
                          if layer['type'] in valid_layer_types]
            print(f"过滤后的模型配置: {model_config}")

            # 构建模型并加载参数
            model = Net(model_config, 
                      input_shape=model_data['input_shape'],
                      num_classes=model_data['num_classes'])
            # 转换状态字典的键名
            state_dict = model_data['model_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                # 将 'model.' 前缀替换为空
                if k.startswith('model.'):
                    new_key = k
                else:
                    new_key = 'model.' + k
                new_state_dict[new_key] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()

            # 处理上传的图片
            img = Image.open(file)
            
            # 将图片转换为base64以便在前端显示
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            last_inference_data['input_image'] = f'data:image/png;base64,{img_str}'

            # 保存原始图像尺寸用于调试
            original_size = img.size
            print(f"原始图像尺寸: {original_size}")

            # 根据数据集类型进行预处理
            if dataset_type in ['MNIST', 'FashionMNIST']:
                # 转换为灰度图
                img = img.convert('L')
                # 调整大小为28x28
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                # 反转颜色（确保白色笔画在黑色背景上）
                img = ImageOps.invert(img)
            elif dataset_type == 'CIFAR10':
                # 转换为RGB
                img = img.convert('RGB')
                # 调整大小为32x32
                img = img.resize((32, 32), Image.Resampling.LANCZOS)

            # 保存处理后的图像尺寸用于调试
            processed_size = img.size
            print(f"处理后图像尺寸: {processed_size}")

            # 构建与训练时相同的预处理流程
            transform_list = []
            
            # 添加数据增强（推理时只使用基本变换）
            if 'preprocessing' in model_data:
                preprocessing = model_data['preprocessing']
                print(f"应用预处理配置: {preprocessing}")
            
            # 基本变换
            transform_list.append(transforms.ToTensor())
            
            # 添加标准化，确保与训练时使用相同的参数
            if model_data['preprocessing'].get('normalization') == 'StandardNormalization':
                if dataset_type == 'CIFAR10':
                    transform_list.append(transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ))
                else:  # MNIST 和 FashionMNIST
                    transform_list.append(transforms.Normalize(
                        mean=[0.5],
                        std=[0.5]
                    ))
            elif model_data['preprocessing'].get('normalization') == 'MinMaxNormalization':
                transform_list.append(transforms.Lambda(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                ))
            
            transform = transforms.Compose(transform_list)
            img_tensor = transform(img).unsqueeze(0)

            # 打印张量信息用于调试
            print(f"输入张量形状: {img_tensor.shape}")
            print(f"输入张量值范围: [{img_tensor.min().item():.2f}, {img_tensor.max().item():.2f}]")

            # 验证张量形状
            expected_shape = (1,) + model_data['input_shape']
            if img_tensor.shape != expected_shape:
                print(f"张量形状不匹配: 期望 {expected_shape}, 实际 {img_tensor.shape}")
                return jsonify({'error': '图像处理后的形状不匹配'}), 400

            # 进行预测
            with torch.no_grad():
                output = model(img_tensor)
                intermediate_outputs = model.get_intermediate_outputs()
                last_inference_data['intermediate_outputs'] = intermediate_outputs
                
                # 收集推理过程的详细信息
                inference_details = []
                # 添加输入信息
                inference_details.append({
                    'type': 'input',
                    'shape': list(img_tensor.shape),
                    'range': [float(img_tensor.min()), float(img_tensor.max())]
                })
                
                # 添加每一层的信息
                for i, (name, layer) in enumerate(zip(model.layer_names, model.model)):
                    layer_info = {
                        'layer_num': i + 1,
                        'name': name,
                        'type': name,
                        'output_shape': list(intermediate_outputs[f'{name}_{i}'].shape),
                        'output_range': [
                            float(intermediate_outputs[f'{name}_{i}'].min()),
                            float(intermediate_outputs[f'{name}_{i}'].max())
                        ]
                    }
                    
                    # 添加层特定的参数
                    if isinstance(layer, nn.Conv2d):
                        layer_info.update({
                            'params': {
                                'in_channels': layer.in_channels,
                                'out_channels': layer.out_channels,
                                'kernel_size': layer.kernel_size[0],
                                'stride': layer.stride[0],
                                'padding': layer.padding[0]
                            }
                        })
                    elif isinstance(layer, nn.MaxPool2d):
                        layer_info.update({
                            'params': {
                                'kernel_size': layer.kernel_size,
                                'stride': layer.stride
                            }
                        })
                    elif isinstance(layer, nn.Linear):
                        layer_info.update({
                            'params': {
                                'in_features': layer.in_features,
                                'out_features': layer.out_features
                            }
                        })
                    
                    inference_details.append(layer_info)

                # 处理中间层输出，转换为可视化数据
                visualization_data = []
                for layer_name, layer_output in intermediate_outputs.items():
                    if layer_output.dim() <= 2:  # 对于全连接层
                        data = {
                            'type': 'dense',
                            'name': layer_name,
                            'shape': list(layer_output.shape),
                            'values': layer_output.numpy().tolist()
                        }
                    else:  # 对于卷积层和池化层
                        data = {
                            'type': 'conv',
                            'name': layer_name,
                            'shape': list(layer_output.shape),
                            'feature_maps': layer_output.numpy().tolist()
                        }
                    visualization_data.append(data)

                # 获取预测概率
                probabilities = F.softmax(output, dim=1)
                pred_prob, pred = probabilities.max(1)
                
                print(f"预测类别: {pred.item()}, 置信度: {pred_prob.item():.2%}")
                
                # 如果置信度太低，可能是预处理问题
                if pred_prob.item() < 0.5:
                    print(f"警告：预测置信度较低 ({pred_prob.item():.2%})")
                    print(f"所有类别的概率: {probabilities[0].tolist()}")

                return jsonify({
                    'prediction': pred.item(),
                    'confidence': float(pred_prob.item()),
                    'dataset': dataset_type,
                    'probabilities': [float(p) for p in probabilities[0].tolist()],
                    'inference_details': inference_details,
                    'visualization_data': visualization_data
                })

        except Exception as e:
            print(f"推理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    # GET 请求返回推理页面
    return render_template('inference.html')

@app.route('/training_progress')
def get_training_progress():
    global training_progress
    with progress_lock:
        return jsonify({'progress': training_progress})

def update_progress(progress):
    global training_progress
    with progress_lock:
        training_progress = progress

@app.route('/generate_examples')
def generate_examples():
    try:
        if not os.path.exists('static/examples'):
            os.makedirs('static/examples/cifar10', exist_ok=True)
            os.makedirs('static/examples/fashion', exist_ok=True)
            os.makedirs('static/examples/mnist', exist_ok=True)

        # 生成CIFAR10示例
        cifar_dataset = datasets.CIFAR10('./data', train=True, download=True)
        class_samples = {i: [] for i in range(10)}
        for img, label in cifar_dataset:
            if len(class_samples[label]) < 1:  # 每个类别取一个样本
                img.save(f'static/examples/cifar10/{label}.png')
                class_samples[label].append(True)
            if all(len(samples) >= 1 for samples in class_samples.values()):
                break

        # 生成Fashion MNIST示例
        fashion_dataset = datasets.FashionMNIST('./data', train=True, download=True)
        class_samples = {i: [] for i in range(10)}
        for img, label in fashion_dataset:
            if len(class_samples[label]) < 1:
                img.save(f'static/examples/fashion/{label}.png')
                class_samples[label].append(True)
            if all(len(samples) >= 1 for samples in class_samples.values()):
                break

        # 生成MNIST示例
        mnist_dataset = datasets.MNIST('./data', train=True, download=True)
        class_samples = {i: [] for i in range(10)}
        for img, label in mnist_dataset:
            if len(class_samples[label]) < 1:
                img.save(f'static/examples/mnist/{label}.png')
                class_samples[label].append(True)
            if all(len(samples) >= 1 for samples in class_samples.values()):
                break

        return jsonify({'status': 'success', 'message': '示例图片生成完成'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_models')
def get_models():
    """获取所有保存的模型列表"""
    try:
        models_dir = 'static/models'
        if not os.path.exists(models_dir):
            return jsonify([])
        
        models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualization/<model_name>')
def visualization(model_name):
    try:
        print(f"Loading visualization for model: {model_name}")
        # 加载模型数据
        model_path = f'static/models/{model_name}'
        model_data = torch.load(model_path)
        print("Model data loaded successfully")
        print("Model config:", model_data['model_config'])
        
        # 获取最后一次推理的数据
        print("Last inference data:", last_inference_data if 'last_inference_data' in globals() else None)
        
        visualization_data = {
            'model_structure': model_data['model_config'],
            'input_image': getattr(last_inference_data, 'input_image', None) if 'last_inference_data' in globals() else None,
            'intermediate_outputs': getattr(last_inference_data, 'intermediate_outputs', {}) if 'last_inference_data' in globals() else {},
            'input_shape': model_data['input_shape'],
            'dataset_type': model_data['dataset']
        }
        
        print("Prepared visualization data:", visualization_data)

        return render_template('visualization.html', 
                            model_name=model_name,
                            dataset=model_data['dataset'],
                            model_config=model_data['model_config'],
                            visualization_data=visualization_data)
    except Exception as e:
        print(f"Error loading visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # 启动时生成示例图片
    try:
        if not os.path.exists('static/examples'):
            print("正在生成示例图片...")
            generate_examples()
            print("示例图片生成完成")
    except Exception as e:
        print(f"生成示例图片时出错: {str(e)}")
    
    app.run(debug=True)
