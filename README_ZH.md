# AI可视化工具链

一个基于Web的深度学习模型可视化工具，支持模型训练、推理和可视化。该工具提供了直观的界面来构建、训练和测试神经网络模型。

## 功能特点

- 支持多种数据集：
  - MNIST（手写数字识别）
  - CIFAR10（物体识别）
  - FashionMNIST（服装分类）
- 可视化模型构建：
  - 拖拽式模型层构建
  - 实时参数配置
  - 模型结构验证
- 模型训练：
  - 支持多种数据增强方法
  - 实时训练进度显示
  - 训练曲线可视化
  - 早停机制
- 模型推理：
  - 支持手绘输入（MNIST）
  - 图片上传功能
  - 示例图片测试
  - 推理过程可视化
- 模型可视化：
  - 3D模型结构展示
  - 特征图可视化
  - 中间层输出展示

## 安装说明

1. 克隆仓库：

```bash
git clone [repository-url]
cd [repository-name]
```

2. 创建并激活虚拟环境（推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

1. 启动应用：

```bash
python app.py
```

2. 打开浏览器访问：

```
http://localhost:5000
```

3. 主要功能页面：

- 首页 (`/`): 模型构建界面
- 训练结果页面 (`/training_results`): 显示训练过程的损失和准确率曲线
- 推理页面 (`/inference`): 模型测试界面
- 可视化页面 (`/visualization/<model_name>`): 模型结构和推理过程可视化

## 项目结构

```
.
├── app.py              # 主应用文件
├── utils.py           # 工具函数和模型定义
├── requirements.txt   # 项目依赖
├── static/           # 静态文件
│   ├── css/         # 样式文件
│   ├── js/          # JavaScript文件
│   ├── models/      # 保存的模型文件
│   └── examples/    # 示例图片
└── templates/        # HTML模板
    ├── index.html    # 首页模板
    ├── train.html    # 训练结果页面
    ├── inference.html # 推理页面
    └── visualization.html # 可视化页面
```

## 开发环境

- Python 3.7+
- PyTorch 1.9+
- Flask 2.0+
- 现代浏览器（支持HTML5和WebGL）

## 注意事项

1. 首次运行时会自动下载数据集，请确保网络连接正常 (项目仓库已包含)
2. 推荐使用支持WebGL的现代浏览器
3. 训练大型模型时可能需要较长时间，请耐心等待
4. 手写输入功能仅支持MNIST数据集

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 致谢

- PyTorch团队
- Flask团队
- Three.js团队
- D3.js团队
