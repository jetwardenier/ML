from perceptron import Perceptron
import unittest

# Perceptron test met weigths 0.1, 0.5 en 0.1 en bias 2

def test():
    p = Perceptron([0.1, 0.5, 0.1],2)
    output = p.activate([0.0,0.2,0.1])
    print(output)
#test()

## Invert(not) gate zet input 1 om in 0 en vice versa
def invert():
    p = Perceptron([-1], 0.5)
    inv1 = p.activate([0])
    inv2 = p.activate([1])
    p = Perceptron([-1], 0)
    print(inv1, inv2)
invert()

## AND gate
def andgate():
    p = Perceptron([0.5, 0.5],-1)
    and1 = p.activate([1,1])
    and2 = p.activate([1,0])
    and3 = p.activate([0,1])
    and4 = p.activate([0,0])
    print(and1, and2, and3, and4)
andgate()

## OR gate (inclusive)
def orgate():
    p = Perceptron([1, 1],-1)
    or1 = p.activate([1, 1])
    or2 = p.activate([1, 0])
    or3 = p.activate([0, 1])
    or4 = p.activate([0, 0])
    print(or1, or2, or3, or4)
orgate()

## NOR gate (overal 0 behalve bij x1 = x2 = x3 =0)
def norgate():
    p = Perceptron([-0.5,-0.5,-0.5], 0.25)
    nor1 = p.activate([1, 1, 1])
    nor2 = p.activate([1, 1, 0])
    nor3 = p.activate([1, 0, 1])
    nor4 = p.activate([1, 0, 0])
    nor5 = p.activate([0, 1, 1])
    nor6 = p.activate([0, 1, 0])
    nor7 = p.activate([0, 0, 1])
    nor8 = p.activate([0, 0, 0])
    print(nor1, nor2, nor3, nor4, nor5, nor6, nor7, nor8)
norgate()

## Party poort 2.8
def partypoort():
    p = Perceptron([0.6, 0.3, 0.2], -0.4)
    party1 = p.activate([1, 1, 1])
    party2 = p.activate([1, 1, 0])
    party3 = p.activate([1, 0, 1])
    party4 = p.activate([1, 0, 0])
    party5 = p.activate([0, 1, 1])
    party6 = p.activate([0, 1, 0])
    party7 = p.activate([0, 0, 1])
    party8 = p.activate([0, 0, 0])
    print(party1, party2, party3, party4, party5, party6, party7, party8)
partypoort()

# def testLayer():
#     l = (5, [0.5, -0.5], 2)
#     l = l.activate_layer([1,1,1])
#     print(l)
# testLayer()