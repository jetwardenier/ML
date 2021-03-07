from perceptron import Perceptron
from perceptronLayer import *
from perceptronNetwork import *
import unittest

# Perceptron test met weigths 0.1, 0.5 en 0.1 en bias 2

def test():
    p = Perceptron([0.1, 0.5, 0.1],2)
    output = p.activate([0.0,0.2,0.1])
    print(output)
#test()

class PerceptronTest(unittest.TestCase):
## Invert(not) gate zet input 1 om in 0 en vice versa
    def test_invert(self):
            p = Perceptron([-1], 0)
            self.assertEqual(p.activate([1]), 0)

            p = Perceptron([-1], 0)
            self.assertEqual(p.activate([0]), 1)

## AND gate
    def test_andgate(self):
        p = Perceptron([0.5, 0.5], -1)
        self.assertEqual(p.activate([0, 0]), 0)

        self.assertEqual(p.activate([1, 0]), 0)

        self.assertEqual(p.activate([0, 1]), 0)

        self.assertEqual(p.activate([1, 1]), 1)

    ## OR gate (inclusive)
    def test_orgate(self):
        p = Perceptron([0.5, 0.5], -0.5)
        self.assertEqual(p.activate([0, 0]), 0)

        self.assertEqual(p.activate([1, 0]), 1)

        self.assertEqual(p.activate([0, 1]), 1)

        self.assertEqual(p.activate([1, 1]), 1)


    ## NOR gate (overal 0 behalve bij x1 = x2 = x3 =0)
    def test_norgate(self):
        p = Perceptron([-1, -1, -1], 0)
        self.assertEqual(p.activate([0, 0, 0]), 1)

        self.assertEqual(p.activate([1, 0, 0]), 0)

        self.assertEqual(p.activate([0, 1, 0]), 0)

        self.assertEqual(p.activate([1, 1, 0]), 0)

        self.assertEqual(p.activate([0, 0, 1]), 0)

        self.assertEqual(p.activate([1, 0, 1]), 0)

        self.assertEqual(p.activate([0, 1, 1]), 0)

        self.assertEqual(p.activate([1, 1, 1]), 0)

    ## Party poort 2.8
    def test_partypoort(self):
        p = Perceptron([0.6, 0.3, 0.2], -0.4)
        self.assertEqual(p.activate([0, 0, 0]), 0)

        self.assertEqual(p.activate([1, 0, 0]), 1)

        self.assertEqual(p.activate([0, 1, 0]), 0)

        self.assertEqual(p.activate([1, 1, 0]), 1)

        self.assertEqual(p.activate([0, 0, 1]), 0)

        self.assertEqual(p.activate([1, 0, 1]), 1)

        self.assertEqual(p.activate([0, 1, 1]), 1)

        self.assertEqual(p.activate([1, 1, 1]), 1)

if __name__ == '__main__':
    unittest.main()