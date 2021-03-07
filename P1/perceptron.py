from typing import List, Sequence
import random
import math


class Perceptron():

    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self.learningRate = 1
        self.n = 0
        self.totalError = 0

        if activation is None:
            activation = self.step
        self.activation = activation

    def step(self, total):
        return 1 if total >= 0 else 0

    def activate(self, _input: List[int]):
        if len(self.weights) != len(_input):
            return None

        total = 0
        for i in range(0, len(self.weights)):
            total += self.weights[i] * _input[i]
        total += self.bias
        return self.activation(total)

    def update(self, _input: List[int], target: int):
        output = self.activate(_input)
        error = target - output
        b_delta = self.learningRate * error

        self.totalError += error
        self.n += 1

        for i in range(len(_input)):
            self.weights[i] += b_delta * _input[i]

        self.bias += b_delta
        return -1 if error == 0 else 0

    def mse(self):
        return (self.totalError ** 2) / self.n

    def __str__(self):
        return f"Weights {self.weights}, Bias {self.bias}, Learningrate {self.learningRate}"

