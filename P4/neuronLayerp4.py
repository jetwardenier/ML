from typing import List
from P4.Neuronp4 import Neuron

class neuronLayer:
    def __init__(self, neurons: [[Neuron]]):
        self.neurons = neurons
        self.errors = []

    def activateLayer(self, input: List[float]) -> list:
        return [i.activate(input) for i in self.neurons]

    ## Calculates the error of the output layer
    def outputError(self, target: List[int or float]) -> None:
        for index, output_neuron in enumerate(self.neurons):
            output_neuron.outputError(output_neuron.output, target[index])
    ## Calculates the error of the hidden layer
    def hiddenError(self, next_layerW:List[List[float]], next_layerE:List[float]) -> None:
        for index, neuron in enumerate(self.neurons):
            next_weight, next_error = [], []
            for weight in next_layerW:
                next_weight.append(weight[index])
            self.neurons[index].hiddenError(neuron.output, next_weight, next_layerE)
            next_error.append(self.neurons[index].error)
            self.errors = next_error

    def update(self, learningRate: float = 0.1) -> None:
        for neuron in self.neurons:
            neuron.update(learningRate)

    def __str__(self):
        string = ""
        for neuron in self.neurons:
            string += "neuron: " + neuron.__str__() + '\n'
        return string