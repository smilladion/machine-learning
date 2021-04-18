from trainers import PyTorchTrainer
import torch
from torchvision import transforms
from networks import Linear, MLP, BasicNetwork, MLP_many_layer, CNN, CNN4Layer, TopCNN


CNN_model1 = CNN()
CNN_Trainer1 = PyTorchTrainer(CNN_model1, transforms.ToTensor(), torch.optim.SGD(CNN_model1.parameters(), lr=0.01), 128)
CNN_Trainer1.train(10)
CNN_Trainer1.save()

CNN_model2 = TopCNN()
CNN_Trainer2 = PyTorchTrainer(CNN_model2, transforms.ToTensor(), torch.optim.SGD(CNN_model2.parameters(), lr=0.01), 128)
CNN_Trainer2.train(10)
CNN_Trainer2.save()

CNN_model3 = CNN4Layer()
CNN_Trainer3 = PyTorchTrainer(CNN_model3, transforms.ToTensor(), torch.optim.SGD(CNN_model3.parameters(), lr=0.01), 128)
CNN_Trainer3.train(10)
CNN_Trainer3.save()

"""
MLP_model1 = MLP()
MLP_Trainer1 = PyTorchTrainer(MLP_model1, transforms.ToTensor(), torch.optim.SGD(MLP_model1.parameters(), lr=0.01), 128)
MLP_Trainer1.train(10)
MLP_Trainer1.save()

MLP_model2 = BasicNetwork()
MLP_Trainer2 = PyTorchTrainer(model2, transforms.ToTensor(), torch.optim.SGD(MLP_model2.parameters(), lr=0.01), 128)
MLP_Trainer2.train(10)
MLP_Trainer2.save()

MLP_model3 = MLP_many_layer()
MLP_Trainer3 = PyTorchTrainer(MLP_model3, transforms.ToTensor(), torch.optim.SGD(MLP_model3.parameters(), lr=0.01), 128)
MLP_Trainer3.train(10)
MLP_Trainer3.save()
"""
