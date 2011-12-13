from numpy import matrix, hstack, multiply, tanh
from numpy.random import standard_normal as randn

# Adds bias to a given input by concatinating a column of 1s
def addBias(Input):
	l = Input.shape[0]
	Bias = matrix([1]*l)
	return hstack((Input,Bias.transpose()))
	
# Runs a single layer of the Artifical Neural Network
def runLayer(I,W):
	S = I*W
	O = tanh(S)
	return O

# Runs the whole network with the given parameters
def runANN(Inputs,Weights):

	# Output of input layer
	Output = [Inputs]	
	# Add bias to the inputs
	I = addBias(Inputs)

	for W in Weights:
		# Run the current layer
		O = runLayer(I,W)
		# Store the current output
		Output.append(O)
		# Add the bias for input of next layer
		I = addBias(O)
	return Output

# Train a Network with the given parameters
def trainANN(Inputs, Targets, Weights, l_min = 0.005, l_max = 0.1, MAX_ITER = 1E4,eps=1E-2):

	# Initialize the parameters
	itr	= 0
	ErrorNorm = []
	num_layers = len(Weights)
	Delta_weights = list(Weights)
	
	# Loop through maximum possible iterations
	while itr < MAX_ITER:
		# Forward Pass, get outputs
		Output = runANN(Inputs,Weights)
		# Generate the error at the output layer
		Error  = Targets - Output[-1]
		# Finding the output norm at the output layer
		ErrorNorm.append(multiply(Error,Error).sum())
		# Break if the error norm is within the limits
		if ErrorNorm[-1] < eps:
			break
		# Backward pass, Error propagation
		i = num_layers-1
		while i > -1:
			# Get the output of the layer
			Layer_Output = Output[i+1]
			# Get the input of the layer
			Layer_Input = addBias(Output[i])
			# Calculating the derivative of the output (1-O^2)
			Deriv = 1 - multiply(Layer_Output,Layer_Output)
			# Calculating the delta value for the current layer
			Delta_value = multiply(Error,Deriv)
			# Calculating the correction weights to be added
			Delta_weights[i] = Layer_Input.transpose()*Delta_value; 
			# Calculating the Error to be back propagated
			Error = Delta_value*Weights[i][:-1].transpose()
			i = i-1
		
		# Backward Pass, Weight Correction
		i = num_layers-1
		# Adaptive learning rate
		learning_rate = l_min + l_max*(1-i/MAX_ITER)
		while i > -1:
			# Update the weights using the learning rate
			Weights[i] = Weights[i]+Delta_weights[i]*learning_rate
			i = i-1
		itr = itr+1
	return Weights,ErrorNorm

# Creating a random ANN with layer parameters
def createANN(layers):
	Weights = []
	l = len(layers)
	for i in range(1,l):
		Weights.append(matrix(randn((layers[i-1]+1,layers[i]))))
	return Weights
