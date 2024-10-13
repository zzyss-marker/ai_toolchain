from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import io
import base64
from utils import Net, train_model, test_model, plot_training_curve
from PIL import Image

app = Flask(__name__)

# 全局变量存储训练曲线数据
training_loss = []
training_accuracy = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户选择的参数
        conv_layers = int(request.form.get('conv_layers'))
        conv_filters = int(request.form.get('conv_filters'))
        kernel_size = int(request.form.get('kernel_size'))
        pool_size = int(request.form.get('pool_size'))
        dense_units = int(request.form.get('dense_units'))
        learning_rate = float(request.form.get('learning_rate'))
        batch_size = int(request.form.get('batch_size'))
        epochs = int(request.form.get('epochs'))

        # 构建模型
        model = Net(conv_layers, conv_filters, kernel_size, pool_size, dense_units)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 数据加载
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        global training_loss, training_accuracy
        training_loss, training_accuracy = train_model(model, train_loader, criterion, optimizer, epochs)

        # 测试模型
        test_accuracy = test_model(model, test_loader)

        # 保存模型
        torch.save(model.state_dict(), 'models/mnist_model.pth')

        return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    # 绘制训练曲线
    loss_img = plot_training_curve(training_loss, 'Loss')
    acc_img = plot_training_curve(training_accuracy, 'Accuracy')

    return render_template('results.html', loss_img=loss_img, acc_img=acc_img)

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    prediction = None
    if request.method == 'POST':
        # 加载模型
        model = Net()
        model.load_state_dict(torch.load('models/mnist_model.pth'))
        model.eval()

        # 处理上传的图片
        file = request.files['image']
        if file:
            img = transforms.functional.to_tensor(Image.open(file).convert('L').resize((28,28)))
            img = img.unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                pred = output.argmax(dim=1, keepdim=True)
                prediction = pred.item()

    return render_template('inference.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
