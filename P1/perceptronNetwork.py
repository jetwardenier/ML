class PerceptronNetwork:

##Het perceptron netwerk maakt gebruik van layers die één of meerdere perceptrons bevatten
    def __init__(self, layers: list):
        self.layers = layers
        self.networkInput = []
        self.output = []

    def setInput(self, networkInput: list):
        self.networkInput = networkInput

##De feedforward methode activeert de lagen met de perceptrons, de output van de eerste laag dient als input
##van de volgende laag tot de laatste laag is bereikt
    def feedForward(self):
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.activate()
                layerInput = layer.output
            self.output = layerInput