class neuronNetwork:

    def __init__(self, layers: list):
        self.layers = layers
        self.networkInput = []
        self.output = []

    def setInput(self, networkInput: list):
        self.networkInput = networkInput

    ## This fucntion makes sure the output of one layer will be the input for the next layer.
    ## in the end the final output will be generated
    def feedForward(self):
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.activate()
                layerInput = layer.output
            self.output = layerInput