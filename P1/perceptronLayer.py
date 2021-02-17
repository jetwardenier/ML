from perceptron import *

## init voegt alle perceptrons toe aan de layer
class PerceptronLayer():
    def __init__(self, weights, bias):
        self.perceptrons = []
        for i in range (0, len(weights)):
            self.perceptrons.append(Perceptron(weights[i], bias[i]))

## activatie functie aanroepen op perceptrons en output appenden aan de outputlayer
    def activateLayer(self, inputLayer: list):
        outputLayer = []
        for i in range(0, len(self.weigths)):
            perceptron = self.perceptrons[i]
            input = inputLayer
            outputLayer.append(perceptron.activate(input))

