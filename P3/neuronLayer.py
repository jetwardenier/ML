class neuronLayer:
    def __init__(self, neurons: list):
        self.neurons = neurons
        self.layerInput = []
        self.output = []

    def setInput(self, layerInput: list):
        self.layerInput = layerInput

    def activate(self):
        if self.layerInput:
            self.output = []
            for neuron in self.neurons:
                neuron.setInput(self.layerInput)
                neuron.activate()
                self.output.append(neuron.output)

    def __str__(self):
        string = ""
        for neuron in self.neurons:
            string += "neurons:"+ neuron.__str__() + '\n'
        return string
