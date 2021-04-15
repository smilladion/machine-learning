from trainers import PyTorchTrainer
import torch
from torch import nn
from torchvision import transforms


class Linear(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784,
                               10)  # Creates a complete linear layer with 784 input features and 10 output features.

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Used to flatten the 28x28 images to 784-dimensional vectors
        return self.layer(x)


model = Linear()
linear = PyTorchTrainer(model, transforms.ToTensor(), torch.optim.SGD(model.parameters(), lr=0.01), 128)
linear.train(10)
linear.save()
