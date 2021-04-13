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
        # The ratio of correct predictions to the total number of test samples.
        a = 0
        for i in range(10):
            a += self.mat[i][i]

        b = 0
        for i in range(10):
            for j in range(10):
                b += self.mat[i][j]

        return a / b

    @property
    def precision(self):
        # The ratio of correct predictions for a certain class to the number of predictions for that class.
        a = np.empty(10)
        for i in range(10):
            a[i] = self.mat[i][i]

        b = np.empty(10)
        for i in range(10):
            x = 0
            for j in range(10):
                x += self.mat[i][j]

            b[i] = x

        return a / b

    @property
    def recall(self):
        # The ratio of correct predictions for a certain class to the number of samples belonging to that class.
        a = np.empty(10)
        for i in range(10):
            a[i] = self.mat[i][i]

        b = np.empty(10)
        for i in range(10):
            x = 0
            for j in range(10):
                x += self.mat[j][i]

            b[i] = x

        return a / b
