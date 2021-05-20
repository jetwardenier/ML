
from datetime import time
from typing import List

class neuronNetwork:

    def __init__(self, layers: list):
        self.layers = layers
        self.networkInput = []
        self.output = []
        self.loss = []

    def feedForward(self, input: List[float]) -> list:
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.activate()
                layerInput = layer.output
            self.output = layerInput

    def MSE(self, inputs:[List[float]], targets:[List[float]]) -> float:
        for i in range(len(inputs)):
            self.feed_forward(inputs[i])
            self.calculate_loss(targets[i])

        total_loss = sum(self.loss) / len(self.loss)
        self.loss = []
        return total_loss

    def loss(self, target: List[int or float], loss_sum: int = 0) -> None:
        for index in range(len(target)):
            loss_sum += (target[index] - self.outputs[index]) ** 2
        self.loss.append(loss_sum / len(target))

    def train(self, inputs: List[List[float]], targets: List[List[float]], learning_rate: float = 0.1, epochs: int = 10000, max_time: int = 200) -> None:
            start_time, epoch = time.time(), 0
            while epochs > epoch and time.time() - start_time < max_time:
                for index, input_list in enumerate(inputs):
                    self.feed_forward(input_list)
                    target = targets[index]

                    for i in range(len(self.layers[::-1])):
                        if i == 0:
                            self.layers[i - 1].error_output(target)
                    else:
                            next_weights = [neuron.weights for neuron in self.layers[i].neurons]
                            next_errors = [neuron.error for neuron in self.layers[i].neurons]
                            self.layers[i - 1].error_hidden(next_weights, next_errors)

                for i in range(len(self.layers[::-1])):
                    self.layers[i - 1].update(learning_rate)
                epoch += 1
            print("total loss", self.MSE(inputs, targets))
            print("epoches", epoch)