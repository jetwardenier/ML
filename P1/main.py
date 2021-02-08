from perceptron import Perceptron

def test():
    p = Perceptron([0.1, 0.5, 0.1],2)
    x = p.activate([0.0,0.2,0.1])
    print(x)
test()

def invert(): 
    p = Perceptron([0], -1)
    x = p.activate([1])
    print(x)
invert()