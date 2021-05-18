import unittest
from P3.Neuron import *
from P3.neuronLayer import *
from P3.neuronNetwork import *

class TestNetwork(unittest.TestCase):
    def testNeuronNetworkHalfAdder(self):
        ## Hidden layer one
        ORNeuron = Neuron(weights=[100, 100], bias=-50)
        NANDNeuron = Neuron(weights=[-80, -80], bias=100)
        ANDNeuron = Neuron(weights=[80, 80], bias=-100)
        inputLayer = neuronLayer(neurons=[ORNeuron, NANDNeuron, ANDNeuron])

        ## Hidden layer two
        ANDNeuron = Neuron(weights=[50, 50, -10], bias=-80)
        extraNeuron = Neuron(weights=[-100, -100, 1000], bias=0)
        hiddenLayer = neuronLayer(neurons=[ANDNeuron, extraNeuron])

        halfAdderNetwork = neuronNetwork(layers=[inputLayer, hiddenLayer])

        inputNetwork = [[[1, 1], [0, 1]],
                        [[1, 0], [1, 0]],
                        [[0, 1], [1, 0]],
                        [[0, 0], [0, 0]]]
        for testInput in inputNetwork:
            halfAdderNetwork.setInput(networkInput=testInput[0])
            halfAdderNetwork.feedForward()

            output = [round(halfAdderNetwork.output[0]), round(halfAdderNetwork.output[1])]
            self.assertEqual(output, testInput[1])

        ## Again we should change the output given in floats to integers to make sure the half adder
        ## will work.

