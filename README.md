# Handwritten Digit Recognition Using PyTorch

## Overview

This project is a beginner-level deep-learning implementation developed as part of practical learning in Artificial Intelligence and Machine Learning.

It demonstrates the use of PyTorch to build, train and evaluate a feedforward neural network for handwritten digit classification using the MNIST dataset.

The implementation provided practical experience with neural network architecture, forward propagation, backpropagation, optimisation and model evaluation.

**Project achievement:** 97.58% test accuracy across 10,000 MNIST test images.


## Project Objective
The objective is to train a neural network to classify handwritten digit images and evaluate its performance on unseen data.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib
- MNIST Dataset

## Model Architecture
The neural network consists of:
- Input layer: 784 features (28 × 28 pixels)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 classes (digits 0–9)

## Training Configuration
- Optimiser: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Training epochs: 5
- Training images: 60,000
- Testing images: 10,000

## Results

**Test Accuracy: 97.58%**

- Correct predictions: 9,758
- Incorrect predictions: 242

The model was evaluated using the MNIST test dataset.

## Project Files
- `Handwritten_Digit_Recognition.ipynb` — notebook containing the implementation and training process.
- `handwritten_digit_model.pth` — saved trained model parameters.


## Reproducing the Trained Model

The trained model parameters are saved in `handwritten_digit_model.pth`.

To load the saved model, recreate the same neural network architecture used during training before loading its parameters.

### Model Loading

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Recreate the original model architecture
class DigitRecognitionModel(nn.Module):
    def __init__(self):
        super(DigitRecognitionModel, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Select the available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Initialise the model
model = DigitRecognitionModel().to(device)

# Load the saved model parameters
state_dict = torch.load(
    "handwritten_digit_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("Trained model loaded successfully!")
```

### Making a Prediction

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image, actual_label = test_dataset[0]

with torch.no_grad():
    image = image.unsqueeze(0).to(device)
    output = model(image)
    predicted_label = output.argmax(dim=1).item()

print("Actual digit:", actual_label)
print("Predicted digit:", predicted_label)
```

The model expects a grayscale image with dimensions of 28 × 28 pixels, using the same tensor conversion as the original MNIST workflow.


## How to Run
1. Open the notebook using Google Colab or Jupyter Notebook.
2. Install PyTorch and Torchvision if they are not already installed.
3. Run the notebook cells to load the dataset, build the model, train it and evaluate its performance.

## Learning Outcomes
- Introduction to artificial neural networks.
- Practical experience with PyTorch.
- Understanding of model training and backpropagation.
- Model evaluation using classification accuracy.

## Prediction Results

![Digit recognition predictions](prediction_results.png)

## Dataset
[MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/)

## Author
**Abdulsamad Bashir**
