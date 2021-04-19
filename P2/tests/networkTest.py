import unittest
from P2.neuronLayer import *
from P2.neuronNetwork import *
from P2.neuron import *


class TestNeuronNetwork(unittest.TestCase):

    def testNeuronNetworkHalfAdder(self):
        ORneuron = Neuron(weights=[100, 100], bias=-50)
        NANDneuron = Neuron(weights=[-80, -80], bias=100)
        ANDneuron = Neuron(weights=[80, 80], bias=-100)
        inputLayer = NeuronLayer(neurons=[ORneuron, NANDneuron, ANDneuron])

        ANDneuron = Neuron(weights=[50, 50, -10], bias=-80)
        newNeuron = Neuron(weights=[-100, -100, 1000], bias=0)
        hiddenLayer = NeuronLayer(neurons=[ANDneuron, newNeuron])

        halfAdderNetwork = NeuronNetwork(layers=[inputLayer, hiddenLayer])

        inputNetwork = [[[0, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 1], [0, 1]], [[1, 0], [1, 1]]]
        for testInput in inputNetwork:
            halfAdderNetwork.setInput(networkInput=testInput[0])

            halfAdderNetwork.feedForward()

            output = [round(halfAdderNetwork.output[0]), round(halfAdderNetwork.output[1])]
            self.assertEqual(output, testInput[1])

        print(halfAdderNetwork)
