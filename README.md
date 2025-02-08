# AI Visualization Toolchain

A web-based deep learning model visualization tool that supports model training, inference, and visualization. This tool provides an intuitive interface for building, training, and testing neural network models.

## Features

- Multiple Dataset Support:
  - MNIST (Handwritten Digit Recognition)
  - CIFAR10 (Object Recognition)
  - FashionMNIST (Clothing Classification)
- Visual Model Building:
  - Drag-and-drop layer construction
  - Real-time parameter configuration
  - Model structure validation
- Model Training:
  - Multiple data augmentation methods
  - Real-time training progress
  - Training curve visualization
  - Early stopping mechanism
- Model Inference:
  - Hand-drawn input support (MNIST)
  - Image upload functionality
  - Example image testing
  - Inference process visualization
- Model Visualization:
  - 3D model structure display
  - Feature map visualization
  - Intermediate layer output display

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your browser and visit:

```
http://localhost:5000
```

3. Main Function Pages:

- Home (`/`): Model building interface
- Training Results (`/training_results`): Displays loss and accuracy curves
- Inference (`/inference`): Model testing interface
- Visualization (`/visualization/<model_name>`): Model structure and inference process visualization

## Project Structure

```
.
├── app.py              # Main application file
├── utils.py           # Utility functions and model definitions
├── requirements.txt   # Project dependencies
├── static/           # Static files
│   ├── css/         # Style sheets
│   ├── js/          # JavaScript files
│   ├── models/      # Saved models
│   └── examples/    # Example images
└── templates/        # HTML templates
    ├── index.html    # Home page template
    ├── train.html    # Training results page
    ├── inference.html # Inference page
    └── visualization.html # Visualization page
```

## Development Environment

- Python 3.7+
- PyTorch 1.9+
- Flask 2.0+
- Modern browser (with HTML5 and WebGL support)

## Important Notes

1. Datasets will be automatically downloaded on first run(The project repository already contains)
2. A modern browser with WebGL support is recommended
3. Training large models may take considerable time
4. Hand-drawing input is only supported for MNIST dataset

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- PyTorch team
- Flask team
- Three.js team
- D3.js team
