import unittest
from P3.neuron import *

class TestNeuron(unittest.TestCase):
    def testANDNeuron(self):
        ANDNeuron = Neuron(weights=[0.5, 0.5], bias=-1)
        ANDinput = [[[1, 1], 1], [[1, 0], 0], [[0, 1], 0], [[0, 0], 0]]

        # for testInput in ANDinput:
        #     ANDNeuron.setInput(neuronInput=testInput[0])
        #     ANDNeuron.activate()
        #     self.assertEqual(ANDNeuron.output, testInput[1])

        ANDNeuron = Neuron(weights=[100, 100], bias=-100)
        for testInput in ANDinput:
            ANDNeuron.setInput(neuronInput=testInput[0])
            ANDNeuron.activate()
            self.assertEqual(int(ANDNeuron.output), testInput[1])

    def testINVERTNeuron(self):
        INVERTNeuron = Neuron(weights=[-1], bias=0)
        INVERTinput = [[[1], 0],[[0], 1]]
        #
        # for testInput in INVERTinput:
        #     INVERTNeuron.setInput(neuronInput=testInput[0])
        #     INVERTNeuron.activate()
        #     self.assertEqual(INVERTNeuron.output, testInput[1])

        INVERTNeuron = Neuron(weights=[-110], bias=100)
        for testInput in INVERTinput:
            INVERTNeuron.setInput(neuronInput=testInput[0])
            INVERTNeuron.activate()
            self.assertEqual(int(INVERTNeuron.output), testInput[1])

    def testORNeuron(self):
            ORNeuron = Neuron(weights=[1, 1], bias=-1)
            ORinput = [[[1, 1], 1],[[1, 0], 1],[[0, 1], 1],[[0, 0], 0]]
            #
            # for testInput in ORinput:
            #     ORNeuron.setInput(neuronInput=testInput[0])
            #     ORNeuron.activate()
            #     self.assertEqual(ORNeuron.output, testInput[1])

            ORNeuron = Neuron(weights=[100, 100], bias=-50)
            for testInput in ORinput:
                ORNeuron.setInput(neuronInput=testInput[0])
                ORNeuron.activate()
                self.assertEqual(int(ORNeuron.output), testInput[1])

    def testNOR(self):
        NORNeuron = Neuron(weights=[-1, -1, -1], bias=0)
        NORinput = [[[0, 0, 0], 1],[[1, 0, 0], 0], [[1, 1, 0], 0],[[1, 1, 1], 0],[[0, 0, 1], 0],[[0, 1, 0], 0],[[0, 1, 1], 0]]

        # for testInput in NORinput:
        #     NORNeuron.setInput(neuronInput=testInput[0])
        #     NORNeuron.activate()
        #     self.assertEqual(NORNeuron.output, testInput[1])

        NORNeuron = Neuron(weights=[-100, -100, -100], bias=100)
        for testInput in NORinput:
            NORNeuron.setInput(neuronInput=testInput[0])
            NORNeuron.activate()
            self.assertEqual(int(NORNeuron.output), testInput[1])