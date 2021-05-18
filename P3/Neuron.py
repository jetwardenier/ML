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

    # The sigmoid function will return a number between 0 and 1 depending on output of the activate function below.
    # If this sum is positive the sigmoid function will return close to 1, if it's negative it will be close to 0.
    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def activate(self):
        ##This will activate the neuron.
        ## Every input will be multiplied by its weight, then the bias will be added to this sum. The output
        ## will be the input for the sigmoid function.
        if self.input:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"