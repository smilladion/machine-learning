import numpy as np
import torch


class MetricLogger:
    def __init__(self, one_hot=True):
        self.mat = np.zeros((10, 10))
        self.one_hot = one_hot

    def log(self, predicted, target):
        if type(predicted) is torch.Tensor:
            predicted = predicted.detach().numpy()
        if type(target) is torch.Tensor:
            target = target.detach().numpy()

        if self.one_hot:
            predicted = np.argmax(predicted, axis=1)

        for pi, ti in zip(predicted, target):
            self.mat[pi, ti] += 1

    def reset(self):
        self.mat = np.zeros(self.mat.shape)


    @property
    def accuracy(self):
        ...

    @property
    def precision(self):
        ...

    @property
    def recall(self):
        ...