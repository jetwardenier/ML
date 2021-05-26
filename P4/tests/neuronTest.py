import unittest
from P4.Neuronp4 import Neuron

class TestNeuron(unittest.TestCase):

    def testANDNeuron(self):
        ## Creating a new neuron
        ANDNeuron = Neuron(weights=[-0.5, 0.5], bias=-1)
        inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
        for input, output in zip(inputs, outputs):
            self.assertNotEqual(output, ANDNeuron.activate(input))

        ## Testing expectedoutput and given output
        for epoch in range(10000):
            for input, output in zip(inputs, outputs):
                n = ANDNeuron.activate(input)
                ANDNeuron.outputError(n, output)
                ANDNeuron.update(n)

        for input, output in zip(inputs, outputs):
            ANDNeuron.activate(input)
            print(ANDNeuron.output, output)
            self.assertAlmostEqual(ANDNeuron.output, output, delta = 0.1)


if __name__ == '__main__':
    unittest.main()