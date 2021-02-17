# import random
class Perceptron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    ## STEP-activatie functie returnt 1 bij input groter dan 0 en anders 0
    def activate(self, _input: list):
        output = 0
        for i in range(0, len(self.weights)):
            output += (self.weights[i] * _input[i])
            # if output + self.bias >= 0:
            #       return 1
            # else:
            #     return 0
        return 1 if output + self.bias >= 0 else 0


        ## String functie
    def __str__(self):
        return f"Weights {self.weights}, Bias {self.bias}"
1