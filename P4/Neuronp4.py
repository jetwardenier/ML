import math

class Neuron:
    def __init__(self, weights: list, bias):

        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None
        self.error = None
        self.newWeights = []
        self.newBias = None
        self.activation = self.sigmoid

    def setInput(self, neuronInput: list):
        if len(neuronInput) != len(self.weights):
            raise Exception("Number of inputs should match the number of weights")
        else:
            self.input = neuronInput

## 1) Determine the error of the output neuron
    def errorOutput(self, expectedOutput, nextNeuronW=[], nextNeuronE=[]):
        if self.output == None:
            raise Exception("Output is missing")
        else:
            if nextNeuronW and nextNeuronE:
                nextSum = 0
                for i in range(len(nextNeuronW)):
                    nextSum += nextNeuronW[i] * nextNeuronE[i]
                self.error = self.output * (1 - self.output) * nextSum
            else:  self.error = self.output * (1-self.output) * -(expectedOutput - self.output)

## 2) Function to calculate gradient
    def gradient(self):
        return self.output * self.error


    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def activate(self):
        ##Inputs are multiplied with their weights and the bias is added
        if self.input == []:
            raise Exception("Input is missing")
        else:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activation(self.output)
        return self.output

    def backProp(self, learningRate):
        if self.error == None:
            raise Exception("Error is missing")
        else:
            self.newWeights = []
            for i in range(len(self.weights)):
                self.newWeights.append(self.weights[i] - learningRate * self.input[i] * self.error)
            self.newBias = self.bias - learningRate * self.error

    ## 5) Updates the weights and biases
    def update(self):
        if self.newBias != None:
            if self.newWeights:
                self.weights = self.newWeights
                self.bias = self.newBias
            else:
                raise Exception("new weights are missing")
        else:
            raise Exception("new bias is missing")
        self.newWeights = []
        self.newBias = None



    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"

