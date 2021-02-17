from perceptron import *
from typing import List
from perceptronLayer import *
class PerceptronNetwork():
    def __init__(self, layers: list, n, weights,biases):
        self.layers : PerceptronLayer = []

    def feedforward(self,input:List[int]):
        for layer in self.layers:
            input = layer.activateLayer(input)
            return input

