import math
class Neuron:
    def __init__(self, weights: list, bias):
        self.activationFunction = self.sigmoid
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

    def setInput(self, neuronInput: list):
        if len(neuronInput) != len(self.weights):
            raise Exception("Aantal inputs moet gelijk zijn aan het aantal weights")
        else:
            self.input = neuronInput

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def activate(self):
        ##Alle inputs worden vermenigvuldigd met zijn weights en de bias wordt daarna verrekend
        if self.input:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"