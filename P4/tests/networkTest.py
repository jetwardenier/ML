import unittest, numpy
from P4.neuronLayerp4 import neuronLayer
from P4.Neuronp4 import Neuron
from P4.neuronNetworkp4 import neuronNetwork

class testNetwork(unittest.TestCase):
    def testXORNetwork(self):
        ## Creating needed neurons and layers
        Neuron1 = Neuron(weights=[0.2, -0.4], bias=0)
        Neuron2 = Neuron(weights=[0.7, 0.1], bias=0)
        inputLayer = neuronLayer([Neuron1, Neuron2])
        Neuron4 = Neuron(weights=[0.6, 0.9], bias=0)
        hiddenLayer = neuronLayer([Neuron4])
        XORNetwork = neuronNetwork(layers=[inputLayer, hiddenLayer])

        ## Define input and target variables
        inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        targets = [[0], [1], [1], [0]]
        for input, output in zip(inputs, targets):
            outputL = XORNetwork.feedForward(input)
            print(outputL)
        XORNetwork.train(inputs, targets)


        ## After training the network
        for input, output in zip(inputs, targets):
            outputL = XORNetwork.feedForward(input)
            print(outputL, output)
            numpy.testing.assert_array_almost_equal(outputL, output, 1)

    def testHalfAdderNetwork(self):
        ## Creating needed neurons and layers
        Neuron1 = Neuron(weights=[0.0, 0.1], bias=0)
        Neuron2 = Neuron(weights=[0.2, 0.3], bias=0)
        Neuron3 = Neuron(weights=[0.4, 0.5], bias=0)
        inputLayer = neuronLayer([Neuron1, Neuron2, Neuron3])

        Neuron4 = Neuron(weights=[0.6, 0.7, 0.8], bias=0)
        Neuron5 = Neuron(weights=[0.9, 1.0, 1.1], bias=0)
        hiddenLayer = neuronLayer([Neuron4, Neuron5])

        HANetwork = neuronNetwork(layers=[inputLayer, hiddenLayer])

        ## Define input and target variables
        inputs = [[0, 1], [1, 1], [1, 0], [0, 0]]
        targets = [[0, 1], [1, 0], [0, 1], [0, 0]]
        for input, output in zip(inputs, targets):
            outputL= HANetwork.feedForward(input)
            print(outputL)
        HANetwork.train(inputs, targets)

        ## After training the network
        for input, output in zip(inputs, targets):
            outputL = HANetwork.feedForward(input)
            print(outputL, output)
            numpy.testing.assert_array_almost_equal(outputL, output, 1)


if __name__ == '__main__':
    unittest.main()