from perceptron import *
import random
import unittest

from sklearn.datasets import load_iris


class PerceptronLearningRuleTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PerceptronLearningRuleTest, self).__init__(*args, **kwargs)
        random.seed(1740188)
        self.maxTries = 1000

    def test_or(self):
        print("OR test")
        n = 3
        ran = [random.randint(-2, 2) for _ in range(n)]
        p = Perceptron([ran[0], ran[1]], ran[2])

        succes = n + 1
        i = 0
        while succes > 0 and i < self.maxTries:
            succes = n + 1
            i += 1

            succes += p.update([0, 0], 0)
            succes += p.update([1, 0], 1)
            succes += p.update([0, 1], 1)
            succes += p.update([1, 1], 1)

        if i == self.maxTries:
            print(f'--- Perceptron fail to converge ---')
            self.fail()

        self.assertEqual(p.activate([0, 0]), 0)
        self.assertEqual(p.activate([1, 0]), 1)
        self.assertEqual(p.activate([0, 1]), 1)
        self.assertEqual(p.activate([1, 1]), 1)

    def test_and(self):
        print("AND test")
        n = 3
        ran = [random.randint(-2, 2) for _ in range(n)]
        p = Perceptron([ran[0], ran[1]], ran[2])

        succes = n + 1
        i = 0
        while succes > 0 and i < self.maxTries:
            succes = n + 1
            i += 1
            succes += p.update([0, 0], 0)
            succes += p.update([1, 0], 0)
            succes += p.update([0, 1], 0)
            succes += p.update([1, 1], 1)

        print(f'Error: {p.mse()}, Iterations: {i}\n{p}')

        if i == self.maxTries:
            print(f'--- Perceptron fail to converge ---')
            self.fail()

        self.assertEqual(p.activate([0, 0]), 0)
        self.assertEqual(p.activate([1, 0]), 0)
        self.assertEqual(p.activate([0, 1]), 0)
        self.assertEqual(p.activate([1, 1]), 1)

    def test_xor(self):
        print("XOR test")
        n = 3
        ran = [random.randint(-2, 2) for _ in range(n)]
        p = Perceptron([ran[0], ran[1]], ran[2])

        succes = n + 1
        i = 0
        while succes > 0 and i < self.maxTries:
            succes = n + 1
            i += 1
            succes += p.update([0, 0], 0)
            succes += p.update([1, 0], 1)
            succes += p.update([0, 1], 1)
            succes += p.update([1, 1], 0)

        print(f'Error: {p.mse()}, Iterations: {i}\n{p}')

        if i == self.maxTries:
            print(f'--- Perceptron fail to converge ---')
            self.fail()

    def test_iris(self):
        print("Iris test gedeelte")
        n = 5
        ran = [random.randint(-2, 2) for _ in range(n)]
        p = Perceptron([ran[0], ran[1], ran[2], ran[3]], ran[-1])
        data = load_iris()

        succes = len(data.data[:100])
        i = 0
        while succes > 0 and i < self.maxTries:
            succes = n + 1
            i += 1
            for i, flwr in enumerate(data.data[:100], 0):
                succes += p.update(flwr, data.target[i])

        print(f'Error: {p.mse()}, Iterations: {i}\n{p}')

        if i == self.maxTries:
            print(f'--- Perceptron fail to converge ---')
            self.fail()

    def test_total_iris(self):
        print("Iris test geheel")
        n = 5
        ran = [random.randint(-2, 2) for _ in range(n)]
        p = Perceptron([ran[0], ran[1], ran[2], ran[3]], ran[-1])
        data = load_iris()

        succes = len(data.data)
        i = 0
        while succes > 0 and i < self.maxTries:
            succes = n + 1
            i += 1
            for i, flwr in enumerate(data.data, 0):
                succes += p.update(flwr, data.target[i])

        print(f'Error: {p.mse()}, Iterations: {i}\n{p}')

        if i == self.maxTries:
            print(f'--- Perceptron fail to converge ---')
            self.fail()


if __name__ == '__main__':
    unittest.main()