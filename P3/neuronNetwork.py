class neuronNetwork:

    def __init__(self, layers: list):
        self.layers = layers
        self.networkInput = []
        self.output = []

    def setInput(self, networkInput: list):
        self.networkInput = networkInput

    def feedForward(self):
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.activate()
                layerInput = layer.output
            self.output = layerInput