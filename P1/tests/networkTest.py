import unittest
from perceptron import *
from perceptronLayer import *
from perceptronNetwork import *

class TestPerceptronNetwork(unittest.TestCase):
    def testXORNetwork(self):
        ##Aanmaken inputlaag
        ORperceptron = Perceptron(weights=[1, 1], bias=-1)
        NANDperceptron = Perceptron(weights=[-1, -1], bias=1)
        inputLayer = PerceptronLayer(perceptrons=[ORperceptron, NANDperceptron])

        ##Aanmaken tweede laag
        ANDperceptron = Perceptron(weights=[1, 1], bias=-2)
        hiddenLayer = PerceptronLayer(perceptrons=[ANDperceptron])

        self.XORnetwork = PerceptronNetwork(layers=[inputLayer, hiddenLayer])

        ##Testen van het netwerk met alle mogelijke inputs
        inputNetwork = [[[1, 1], 0],[[1, 0], 1],[[0, 1], 1],[[0, 0], 0]]

        for testInput in inputNetwork:
            self.XORnetwork.setInput(networkInput=testInput[0])

            #Aanroepen feedforward methode
            self.XORnetwork.feedForward()

            #Vergelijk om te checken of de juiste output gegeven wordt
            self.assertEqual(self.XORnetwork.output[0], testInput[1])

    def testHalfAdder(self):
        ORperceptron = Perceptron(weights=[1, 1], bias=-1)
        NANDperceptron = Perceptron(weights=[-1, -1], bias=1)
        ANDperceptron = Perceptron(weights=[1, 1], bias=-2)
        inputLayer = PerceptronLayer(perceptrons=[ORperceptron, NANDperceptron, ANDperceptron])

        ANDperceptron = Perceptron(weights=[1, 1, 0], bias=-2)
        newPerceptron = Perceptron(weights=[0, 0, 1], bias=-1)
        hiddenLayer = PerceptronLayer(perceptrons=[ANDperceptron, newPerceptron])

        self.halfAdderNetwork = PerceptronNetwork(layers=[inputLayer, hiddenLayer])

        ##Testen van het netwerk met alle mogelijke inputs
        inputNetwork = [[[1, 1], [0, 1]],[[1, 0], [1, 0]], [[0, 1], [1, 0]],[[0, 0], [0, 0]]]

        for testInput in inputNetwork:
            self.halfAdderNetwork.setInput(networkInput=testInput[0])
            self.halfAdderNetwork.feedForward()

            # Vergelijk om te checken of de juiste output gegeven wordt
            self.assertEqual(self.halfAdderNetwork.output, testInput[1])