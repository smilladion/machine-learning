from trainers import PyTorchTrainer
import torch
from torchvision import transforms
from networks import Linear


model = Linear()
linear = PyTorchTrainer(model, transforms.ToTensor(), torch.optim.SGD(model.parameters(), lr=0.01), 128)
linear.train(10)
linear.save()
