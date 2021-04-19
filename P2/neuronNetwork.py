class NeuronNetwork:
    def __init__(self, layers: list):
        self.layers = layers
        self.networkInput = []
        self.output = []

    def setInput(self, networkInput: list):
        self.networkInput = networkInput

    def feedForward(self):
        if self.networkInput:
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.activate()
                layerInput = layer.output
            self.output = layerInput

    def __str__(self):
        string = ''
        for i in range(len(self.layers)):
            string += f"layer: {i}: \n {self.layers[i].__str__()} \n"
        return string