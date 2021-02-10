from perceptron import Perceptron
# Perceptron test met weigths 0.1, 0.5 en 0.1 en bias 2 

def test():
    p = Perceptron([0.1, 0.5, 0.1],2)
    output = p.activate([0.0,0.2,0.1])
    print(output)
#test()

## Invert(not) gate zet input 1 om in 0 en vice versa
def invert(): 
    p = Perceptron([-1], 0.5)
    output = p.activate([0])
    print(output)
#invert()

## AND gate
def andgate():
    p = Perceptron([1, 1],1)
    output = p.activate([1,0])
    print(output)
andgate()

## OR gate (inclusive)
def orgate():
    p = Perceptron([0.5, 0.5],-0.25)
    output = p.activate([1,0])
    print(output)
#orgate()

## NOR gate (overal 0 behalve bij x1 = x2 = x3 =0)
def norgate():
    p = Perceptron([-0.5,-0.5,-0.5], 0.25)
    output = p.activate([0,0,0])
    print(output)
#norgate()

## Party poort 2.8
def partypoort():
    p = Perceptron([0.6, 0.3, 0.2], 0.4)
    output = p.activate([0,0,0])
    print(output)
#partypoort()

