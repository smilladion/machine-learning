import os
from abc import ABC, abstractmethod, abstractstaticmethod

from datetime import datetime

import pickle
from typing import Any

import torch
import torch.nn.functional as F

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from fashionmnist_utils import mnist_reader

from metrics import MetricLogger


class Trainer(ABC):
    def __init__(self, model):
        self.model = model
        self.name = (
            f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'
        )

    @abstractmethod
    def train(self, *args):
        ...

    @abstractmethod
    def predict(self, input):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        ...


class SKLearnTrainer(Trainer):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        X, y = mnist_reader.load_mnist("FashionMNIST/raw", kind="train")

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_test, self.y_test = mnist_reader.load_mnist(
            "FashionMNIST/raw", kind="t10k"
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

        y_predict = self.model.predict(self.X_val)

        metric = MetricLogger(False)
        metric.log(y_predict, self.y_val)

        print("Accuracy: " + str(metric.accuracy))
        print("Precision: " + str(metric.precision))
        print("Recall: " + str(metric.recall))

    def predict(self, input):
        return self.model.predict(input)

    def evaluate(self):
        y_predict = self.model.predict(self.X_test)

        metric = MetricLogger(False)
        metric.log(y_predict, self.y_test)

        return metric

    def save(self):
        with open(os.path.join("models", self.name + ".pkl"), "wb") as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load(path: str):
        new = SKLearnTrainer(None)
        with open(path, "rb") as file:
            new.model = pickle.load(file)
            new.name = os.path.basename(path).split(".")[0]
            return new


def get_data(transform, train=True):
    return FashionMNIST(os.getcwd(), train=train, transform=transform, download=True)


class PyTorchTrainer(Trainer):
    def __init__(self, nn_module, transform, optimizer, batch_size):
        super().__init__(nn_module)

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.init_data()

        self.logger = SummaryWriter()

    def init_data(self):
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)

        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)

    def train(self, epochs):
        train_log = MetricLogger(False)
        val_log = MetricLogger(False)

        for e in range(epochs):

            val_log.reset()

            for i, (x, y) in enumerate(self.train_data):
                out = self.model(x)  # The model instance can be called like a function
                loss = F.cross_entropy(out, y)

                self.optimizer.zero_grad()  # The gradient of the optimizer has to be reset before calculating the gradient.
                loss.backward()
                self.optimizer.step()  # This call updates the model weights.

                y_predict = self.predict(x)
                train_log.log(y_predict, y)

                step = i + e * len(self.train_data)  # Only using i won't work because it resets every epoch

                if i % 100 == 0:
                    self.logger.add_scalar("Accuracy - Training", train_log.accuracy, step)
                    self.logger.add_scalar("Loss - Training", loss, step)
                    train_log.reset()

            for i, (x, y) in enumerate(self.val_data):
                y_predict = self.predict(x)
                val_log.log(y_predict, y)

            self.logger.add_scalar("Accuracy - Validation", val_log.accuracy, e)

    def predict(self, input):
        input = torch.tensor(input).float()
        return torch.argmax(self.model(input), dim=1)

    def evaluate(self):
        metric = MetricLogger(False)

        for i, (x, y) in enumerate(self.test_data):
            y_predict = self.predict(x)
            metric.log(y_predict, y)

        return metric

    def save(self):
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join("models", self.name)
        with open(file_name + ".pkl", "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as file:
            new = pickle.load(file)
            new.init_data()
            return new