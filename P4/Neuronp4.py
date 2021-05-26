from typing import List
import math

class Neuron:
    def __init__(self, weights: List[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.output = None
        self.error = 0
        self.input = []

    def setInput(self, neuronInput: list):
        if len(neuronInput) != len(self.weights):
            raise Exception("Aantal inputs moet gelijk zijn aan het aantal weights")
        else:
            self.input = neuronInput

    def sigmoid(self, x):
            return 1 / (1 + math.e ** -x)

    def activate(self, input: List[float]) -> float:
        ##This will activate the neuron.
        ## Every input will be multiplied by its weight, then the bias will be added to this sum. The output
        ## will be the input for the sigmoid function.
        self.setInput(input)
        self.input = input
        output = 0
        for i in range(len(self.weights)):
            output += input[i] * self.weights[i]
        output += self.bias
        output = self.sigmoid(output)
        self.output = output
        return output

    #1)calculates the error of the output neuron
    def outputError(self, output: float,  expected_output: int) -> float:
        error = self.derivative(output) * -(expected_output - output)
        self.error = error
        return self.error

    #6)calculates the error of the hidden neuron
    def hiddenError(self, output: float, next_weights: List[float], next_errors: List[float]) -> float:
        sum_error = 0
        for index in range(len(next_weights)):
            sum_error += next_weights[index] * next_errors[index]
        self.error = self.derivative(output) * sum_error
        return self.error

    def derivative(self, output: float) -> float:
        return output * (1 - output)

    #5)setting the new values to weights and bias
    def update(self, learning_rate: float = 0.1) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= (learning_rate * self.error * self.input[i])
        self.bias -= (learning_rate * self.error)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}," \
               f" error: {self.error} "