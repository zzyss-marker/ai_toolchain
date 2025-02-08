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
import os
import json
import threading

app = Flask(__name__)

# 全局变量存储训练曲线数据
training_loss = []
training_accuracy = []

# 全局变量存储训练进度
training_progress = 0
progress_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_components', methods=['GET'])
def get_components():
    components = {
        'datasets': ['MNIST'],
        'modules': ['Conv2D', 'MaxPool2D', 'ReLU', 'Flatten', 'Linear'],
        'saved_models': os.listdir('static/models') if os.path.exists('static/models') else []
    }
    return jsonify(components)

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        model_config = data['model_config']
        training_params = data['training_params']

        if not model_config:
            return jsonify({'status': 'error', 'message': '请先添加模型层'}), 400

        # 重置进度
        global training_progress
        training_progress = 0

        # 构建模型
        model = Net(model_config)
        optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # 数据加载
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)

        # 训练模型
        global training_loss, training_accuracy
        training_loss, training_accuracy = train_model(model, train_loader, criterion, optimizer, 
                                                     training_params['epochs'], update_progress)

        # 测试模型
        test_acc = test_model(model, test_loader)

        # 保存模型和配置
        if not os.path.exists('static/models'):
            os.makedirs('static/models')
        model_name = 'model_{}.pth'.format(len(os.listdir('static/models')))
        torch.save(model.state_dict(), 'static/models/' + model_name)
        
        # 保存模型配置
        with open(f'static/models/{model_name}_config.json', 'w') as f:
            json.dump(model_config, f)

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

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        try:
            file = request.files['image']
            model_name = request.form.get('model_name')

            # 加载模型配置
            with open(f'static/models/{model_name}_config.json', 'r') as f:
                model_config = json.load(f)

            # 构建模型并加载参数
            model = Net(model_config)
            model.load_state_dict(torch.load('static/models/' + model_name))
            model.eval()

            # 处理上传的图片
            img = Image.open(file).convert('L')
            # 确保图像是28x28
            if img.size != (28, 28):
                img = img.resize((28, 28))
            
            # 转换为张量并归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
            ])
            
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)

            # 进行预测
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1, keepdim=True)
                prediction = pred.item()

            return jsonify({'prediction': prediction})
        except Exception as e:
            print(f"Inference error: {str(e)}")
            return jsonify({'error': str(e)}), 500

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

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
