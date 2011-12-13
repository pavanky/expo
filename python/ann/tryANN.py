#!/usr/bin/python
from annalgos import *

Inputs = matrix([
		[1,-1,-1],
		[1,-1,1],
		[-1,-1,-1],
		[-1,1,-1],
		[1,1,-1]
		])
Targets = matrix([
		[-1],
		[1],
		[-1],
		[-1],
		[1]
		])
		
x = Inputs.shape[1]
layers = [x,3,2,1]

Weights = createANN(layers)
Weights, Error = trainANN(Inputs, Targets, Weights)
Outputs = runANN(Inputs, Weights)

print(repr(Inputs))
print(repr(Targets))
print(repr(Outputs[-1]))
