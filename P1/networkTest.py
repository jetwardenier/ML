from perceptron import *
import unittest
from perceptronNetwork import *
from perceptronLayer import *

class PerceptronNetworkTest(unittest.TestCase):
    def test_n_xor(self):
        net = PerceptronNetwork(2,  # Layer depth (no input layer needed).
                                [2, 1],  # Nummer of perceptrons
                                [[[1, 1], [1, 1]], [[2, -1]]],  # Incoming Weights per perceptron in the n'th layer.
                                [[-1, -2], [-2]])  # Bias for every perceptron in n'th layer

        self.assertEqual(net.feedforward([1, 1]), [0])
        self.assertEqual(net.feedforward([1, 0]), [1])
        self.assertEqual(net.feedforward([0, 1]), [1])
        self.assertEqual(net.feedforward([0, 0]), [0])

if __name__ == '__main__':
    unittest.main()