from trainers import PyTorchTrainer
import torch
from torchvision import transforms
from networks import Linear, MLP, BasicNetwork, MLP_many_layer


MLP_model1 = MLP()
Trainer1 = PyTorchTrainer(MLP_model1, transforms.ToTensor(), torch.optim.SGD(MLP_model1.parameters(), lr=0.01), 128)
Trainer1.train(10)
Trainer1.save()

MLP_model2 = BasicNetwork()
Trainer2 = PyTorchTrainer(model2, transforms.ToTensor(), torch.optim.SGD(MLP_model2.parameters(), lr=0.01), 128)
Trainer2.train(10)
Trainer2.save()

MLP_model3 = MLP_many_layer()
Trainer3 = PyTorchTrainer(MLP_model3, transforms.ToTensor(), torch.optim.SGD(MLP_model3.parameters(), lr=0.01), 128)
Trainer3.train(10)
Trainer3.save()
