# import random
class Perceptron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, __input:list): 
        for i in range(0, len(self.weights)):
            x = 0
            x += self.weights[i] * __input[i] + self.bias
        if x >= 0:
            return 1
        else:
            return 0
    ## Step activatie functie 

    def __str__(self): 
        return 'weights' + {self.weights} + 'bias' + {self.bias}

# class PerceptronLayer():
#     def __init__(self, list)