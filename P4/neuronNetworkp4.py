
from P4.neuronLayerp4 import neuronLayer
import time
from typing import List

class neuronNetwork:
    def __init__(self, layers: [[neuronLayer]]):
        self.layers = layers
        self.outputs = []
        self.loss = []

    ## This fucntion makes sure the output of one layer will be the input for the next layer.
    ## in the end the final output will be generated
    def feedForward(self, input: List[float]) -> list:
        self.outputs = [input]
        for i in self.layers:
            self.outputs.append(i.activateLayer (list((self.outputs[-1]))))
        self.outputs = self.outputs[-1]
        return self.outputs


    ## Calculates loss for one trainingsexample
    def loss(self, target: List[int or float], loss_sum: int = 0) -> None:
        for index in range(len(target)):
            loss_sum += (target[index] - self.outputs[index])**2
        self.loss.append(loss_sum/len(target))

    ## Calculates total loss for the complete network
    def MSE(self, inputs:[List[float]], targets:[List[float]]) -> float:
        for i in range(len(inputs)):
            self.feedForward(inputs[i])
            self.loss(targets[i])

        totalLoss = sum(self.loss) / len(self.loss)
        self.loss = []
        return totalLoss

    def train(self, inputs:List[List[float]], targets:List[List[float]], learning_rate:float = 0.1,
              epoches:int=10000, max_time:int=200) -> None:
        start_time, epoch = time.time(), 0
        while epoches > epoch and time.time()-start_time < max_time:
            for index, input_list in enumerate(inputs):
                self.feedForward(input_list)
                target = targets[index]

                for i in range(len(self.layers[::-1])):
                    if i == 0:self.layers[i-1].outputError(target)
                    else:
                        next_weights = [neuron.weights for neuron in self.layers[i].neurons]
                        next_errors = [neuron.error for neuron in self.layers[i].neurons]
                        self.layers[i-1].hiddenError(next_weights, next_errors)

                for i in range(len(self.layers[::-1])):
                    self.layers[i-1].update(learning_rate)
            epoch += 1
