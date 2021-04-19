class Perceptron:

    def __init__(self, weights: list, bias: int or float):
        self.activationFunction = self.stepFunction
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

##De input moet uit even veel waardes bestaan als de weights
    def setInput(self, perceptronInput: list):
        if len(perceptronInput) != len(self.weights):
            raise Exception("Aantal inputs moet gelijk zijn aan het aantal weights")
        else:
            self.input = perceptronInput

##Bij alle getallen onder de 0 geeft de stepfunctie 0 terug, bij 0 of groter dan 0 geeft de functie 1 terug
    def stepFunction(self, x):
        return 0 if x < 0 else 1

##De input wordt vermenigvuldigd met zijn weights en de bias wordt daarna verrekend
    def activate(self):
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"