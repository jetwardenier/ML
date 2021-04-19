import unittest
from perceptron import *

class TestPerceptron(unittest.TestCase):

##De AND gate wordt getest met alle mogelijke inputs, de test slaagt als de output overal 0 is
##behalve als beide inputs 1 zijn.
    def testAND(self):
        ANDperceptron = Perceptron(weights=[0.5, 0.5], bias=-1)
        ANDinput = [[[1, 1], 1],[[1, 0], 0],[[0, 1], 0],[[0, 0], 0]]

        for testInput in ANDinput:
            ANDperceptron.setInput(perceptronInput=testInput[0])
            ANDperceptron.activate()
            self.assertEqual(ANDperceptron.output, testInput[1])
##De INVERT gate wordt getest met alle mogelijke inputs, de test slaagt als input 1 output 0 geeft
## en vice versa.
    def testINVERT(self):
        INVERTperceptron = Perceptron(weights=[-1], bias=0)
        INVERTinput = [[[1], 0],[[0], 1]]

        for testInput in INVERTinput:
            INVERTperceptron.setInput(perceptronInput=testInput[0])
            INVERTperceptron.activate()
            self.assertEqual(INVERTperceptron.output, testInput[1])
##De OR gate wordt getest met alle mogelijke inputs, de test slaagt als minstens één input 1 is.
    def testOR(self):
        ORperceptron = Perceptron(weights=[1, 1], bias=-1)

        ORinput = [[[1, 1], 1],[[1, 0], 1],[[0, 1], 1],[[0, 0], 0]]

        for testInput in ORinput:
            ORperceptron.setInput(perceptronInput=testInput[0])
            ORperceptron.activate()
            self.assertEqual(ORperceptron.output, testInput[1])
##De NOR gate wordt getest met alle mogelijke inputs, de output is alleen 1 wanneer alle inputs 0 zijn.
    def testNOR(self):
        NORperceptron = Perceptron(weights=[-1, -1, -1], bias=0)
        NORinput = [[[0, 0, 0], 1],[[1, 0, 0], 0], [[1, 1, 0], 0],[[1, 1, 1], 0],[[0, 0, 1], 0],[[0, 1, 0], 0],[[0, 1, 1], 0]]

        for testInput in NORinput:
            NORperceptron.setInput(perceptronInput=testInput[0])
            NORperceptron.activate()
            self.assertEqual(NORperceptron.output, testInput[1])

##De Party gate wordt getest met alle mogelijke inputs, de weights en outputs zijn genoteerd in de reader onder
##figuur 2.8
    def testPARTY(self):
        PARTYperceptron = Perceptron(weights=[0.6, 0.3, 0.2], bias=-0.4)

        PARTYinput = [[[0, 0, 0], 0],[[1, 0, 0], 1], [[1, 1, 0], 1],[[1, 1, 1], 1],[[0, 1, 1], 1],[[0, 0, 1], 0],[[0, 1, 0], 0]]

        for testInput in PARTYinput:
            PARTYperceptron.setInput(perceptronInput=testInput[0])
            PARTYperceptron.activate()
            self.assertEqual(PARTYperceptron.output, testInput[1])


if __name__ == '__main__':
    unittest.main()