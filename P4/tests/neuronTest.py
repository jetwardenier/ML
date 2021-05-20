import unittest
from P4.Neuronp4 import *


class testNeuron(unittest.TestCase):
    def testANDgate(self):
        truthTable = [
            [[0, 0], 0],
            [[1, 0], 0],
            [[0, 1], 0],
            [[1, 1], 1],
        ]
        andNeuron = Neuron(weights=[-0.5, 0.5], bias=1.5)

        # Laten zien dat de andNeuron nu nog niet werkt
        expectedOutput = [row[1] for row in truthTable]
        output = []
        for row in truthTable:
            andNeuron.setInput(row[0])
            output.append(andNeuron.activate())
        for i in range(len(expectedOutput)):
            self.assertNotAlmostEqual(expectedOutput[i], output[i], delta=0.1)

        # De and neuron trainen
        for epoch in range(10000):
            for row in truthTable:
                andNeuron.setInput(row[0])
                andNeuron.activate()
                andNeuron.errorOutput(row[1])
                andNeuron.backProp(1)
                andNeuron.update()

        # Laten zien dat de end neuron nu wel werkt
        output = []
        for row in truthTable:
            andNeuron.setInput(row[0])
            output.append(andNeuron.activate())
        for i in range(len(expectedOutput)):
            self.assertAlmostEqual(expectedOutput[i], output[i], delta=0.1)


if __name__ == '__main__':
    unittest.main()


    class TestNeuron(unittest.TestCase):

        def test_AND(self):
            """
            Hier wordt de werking van een AND gate getest
            """
            # Maak de neuron aan
            p1 = Neuron(weights=[-0.5, 0.5], bias=-1.5)
            # Maak de inputs en de outputs aan
            inputs, outputs = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1]
            # Vergelijk de output met de verwachte output
            for input, output in zip(inputs, outputs):
                self.assertNotEqual(output, p1.predict(input))

            for _ in range(10000):
                for input, output in zip(inputs, outputs):
                    n = p1.predict(input)
                    p1.cal_error_output(n, output)
                    p1.update(n)
            print(p1)

            for input, output in zip(inputs, outputs):
                p1.predict(input)
                print(p1.antwoord, output)
                self.assertAlmostEqual(p1.antwoord, output, delta=0.1)


    if __name__ == '__main__':
        unittest.main()