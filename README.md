# Simple ResNet Implementation for CIFAR-10

A clean and simple implementation of ResNet architecture for CIFAR-10 classification using PyTorch. This implementation focuses on clarity and simplicity while maintaining good performance.

## Architecture Overview

The network consists of:
- Initial convolution layer
- 3 residual blocks with skip connections
- Global average pooling
- Final fully connected layer

Features:
- Projection shortcuts (1x1 convolutions) for dimension matching
- Batch normalization after each convolution
- ReLU activation functions
- No inplace operations for stable gradient computation

## Requirements

```python
torch>=1.7.0
torchvision>=0.8.0
matplotlib>=3.3.0
```

## Usage

```python
# Import required libraries
import torch
import torchvision
import torchvision.transforms as transforms

# Load and prepare CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Create model instance
model = ResNet().to(device)

# Train the model
train_losses, test_losses = train(model, epochs=100)
```

## Training Details

- Optimizer: SGD with momentum
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- Learning rate schedule: MultiStepLR with milestones at [30, 60, 90]
- Batch size: 128 for training, 32 for testing

## Results

The model achieves:
- Training accuracy: ~90%
- Test accuracy: ~85%
- Convergence in approximately 100 epochs

## Model Architecture Details

```
Input (3, 32, 32)
↓
Conv2d(3 → 16)
↓
3 Residual Blocks with channels: 16 → 32 → 64
↓
Global Average Pooling
↓
Linear(64 → 10)
```

## License

MIT

## Acknowledgments

Based on the ResNet paper: "Deep Residual Learning for Image Recognition" by Kaiming He et al.
