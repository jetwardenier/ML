import unittest
from neuron import *
class TestNeuron(unittest.TestCase):
    def testAND(self):
        ANDneuron = Neuron(weights=[0.5, 0.5], bias=-1)
        ANDinput = [[[0, 0], 0],[[0, 1], 0],[[1, 0], 0],[[1, 1], 1]]
        for testInput in ANDinput:
            ANDneuron.setInput(neuronInput=testInput[0])
            ANDneuron.activate()
            self.assertNotEqual(ANDneuron.output, testInput[1])

        ANDneuron = Neuron(weights=[100, 100], bias=-100)
        for testInput in ANDinput:
            ANDneuron.setInput(neuronInput=testInput[0])
            ANDneuron.activate()
            self.assertEqual(int(ANDneuron.output), testInput[1])

    def testINVERT(self):
        NOTneuron = Neuron(weights=[-1], bias=0)
        NOTinput = [[[1], 0],[[0], 1]]
        for testInput in NOTinput:
            NOTneuron.setInput(neuronInput=testInput[0])
            NOTneuron.activate()
            self.assertNotEqual(NOTneuron.output, testInput[1])

        NOTneuron = Neuron(weights=[-100], bias=50)
        for testInput in NOTinput:
            NOTneuron.setInput(neuronInput=testInput[0])
            NOTneuron.activate()
            self.assertEqual(int(NOTneuron.output), testInput[1])

    def testOR(self):
        ORneuron = Neuron(weights=[0.5, 0.5], bias=-1)
        ORinput = [[[0, 0], 0],[[0, 1], 1],[[1, 0], 1],[[1, 1], 1]]

        for testInput in ORinput:
            ORneuron.setInput(neuronInput=testInput[0])
            ORneuron.activate()
            self.assertNotEqual(ORneuron.output, testInput[1])

        ORneuron = Neuron(weights=[100, 100], bias=-50)
        for testInput in ORinput:
            ORneuron.setInput(neuronInput=testInput[0])
            ORneuron.activate()
            self.assertEqual(int(ORneuron.output), testInput[1])

    def testNeuronNORGATE(self):
        NORneuron = Neuron(weights=[-100, -100, -100], bias=100)
        NORinput = [[[0, 0, 0], 1], [[1, 0, 0], 0],[[1, 1, 0], 0],[[1, 1, 1], 0],[[0, 1, 1], 0],[[0, 0, 1], 0],[[0, 1, 0], 0]]

        for testInput in NORinput:
            NORneuron.setInput(neuronInput=testInput[0])
            NORneuron.activate()
            self.assertEqual(int(NORneuron.output), testInput[1])