class PerceptronLayer:
    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons
        self.layerInput = []
        self.output = []

    def setInput(self, layerInput: list):
        self.layerInput = layerInput

    def activate(self):
        if self.layerInput:
            self.output = []  # Reset the output
            for perceptron in self.perceptrons:
                perceptron.setInput(self.layerInput)
                perceptron.activate()
                self.output.append(perceptron.output)

    def __str__(self):
        string = ""
        for perceptron in self.perceptrons:
            string += perceptron.__str__() + '\n'
        return string
