# import random
class Perceptron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    

## STEP-activatie functie returnt 1 bij input groter dan 0 en anders 0 
    def activate(self, input:list): 
        for i in range(0, len(self.weights)):
            output = 0
            output = (self.weights[i] * self.input[i])
            output = output + self.bias
        if output <= 0:
            return 0
        else:
            return 1

            
# ## String functie
#     def __str__(self): 
#         return 'weights:' {self.weights} 'bias:' {self.bias}

# __str__()

class PerceptronLayer():
    def __init__(self, n, weights, bias):
        self.layer = []
        for i in n: 
            self.layer.append(Perceptron())

    def activate_layer()
        for i in range(0, len(Perceptron.weights)):
            outputlayer = 0
            outputlayer = (Perceptron.weights[i] * Perceptron.input[i])
            outputlayer = outputlayer + perceptron.bias
        if outputlayer <= 0: 
            return 0
        else: 
            return 1

class PerceptronNetwork(): 
    def __init__(self, n, weights, bias)
        self.network = []
        for i in n: 
            self.network.append(PerceptronLayer())

    def feedforward()
        for i in range(0, len(PerceptronLayer.weights)):
            outputnetwork = 0
            outputnetwork = (PerceptronLayer.weights[i] * Perceptron.input[i])
            outputnetwork = outputnetwork + self.bias
        if outputnetwork <= 0: 
            return 0
        else return 1