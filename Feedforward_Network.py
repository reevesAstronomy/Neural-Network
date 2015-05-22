import pickle
import numpy as np
from numpy import random
import matplotlib.pyplot as mpl

#Activation functions and their derivatives (defined here - makes it easier to change them later):
def hidden_act_function(x):
	return np.tanh(x)
def d_hidden_act_function(x):
	return (1.0-x**2)

def output_act_function(x):
	return (1.0/(1.0+np.exp(-x)))
def d_output_act_function(x):
	#return ((output_act_function(x))*(1.0 - output_act_function(x)))
	return x*(1.0-x)

class Network:
	def __init__(self, N_inputs, N_hidden, N_outputs):
		self.N_inputs = N_inputs + 1 #Add 1 to act give a weighted bias vector.
		self.N_hidden = N_hidden
		self.N_outputs = N_outputs

		self.weights1 = random.rand(self.N_hidden, self.N_inputs) #Weights for the connections between the inputs and the hidden layer.
		self.weights2 = random.rand(self.N_outputs, self.N_hidden) #Weights for the connections between the hidden layer and the output(s).

		self.activations_input = np.zeros(self.N_inputs) + 1
		self.activations_hidden = np.zeros(self.N_hidden)
		self.activations_output = np.zeros(self.N_outputs)


	def Classify(self, inputs):
		#Takes in an input array of length = N_inputs.
		for i in np.arange(self.N_inputs-1):
			self.activations_input[i] = inputs[i]

		#Hidden:
		for i in np.arange(self.N_hidden):
			temp = 0.0
			for j in np.arange(self.N_inputs):
				temp += self.activations_input[j]*self.weights1[i][j]
			self.activations_hidden[i] = hidden_act_function(temp)

		#Outputs
		for i in np.arange(self.N_outputs):
			temp = 0.0
			for j in np.arange(self.N_hidden):
				temp += self.activations_hidden[j]*self.weights2[i][j]
			self.activations_output[i] = output_act_function(temp)

		return self.activations_output #Note: This output is NOT a binary value, the ClassifyPoint function handles that.


	def learn(self, vectors, labels, iterations):
		#Runs for the given number of iterations
		length = len(labels)
		pos = 0

		#Use Classify() for each point and then BackPropogate() to update the weight values for each iteration.
		for i in np.arange(iterations):
			for pos in np.arange(length):
				print labels[pos]
				Error = self.BackPropogate(self.Classify(vectors[:,pos]), labels[pos])


	def BackPropogate(self, classifieds, targets):
		#Use backpropagation to update the weight values. Also, calculate error while we're at it.
		print classifieds, targets
		learning_rate = 0.1
		length = len(targets)
		Error = 0.0

		output_error = (targets - self.activations_output)*d_output_act_function(self.activations_output)

		#hidden deltas:
		hidden_deltas = np.zeros(self.N_hidden)
		for i in np.arange(self.N_hidden):
			hidden_error_temp = 0.0
			for j in np.arange(self.N_outputs):
				hidden_error_temp += output_error[j]*self.weights2[j][i]
			hidden_deltas[i] = d_hidden_act_function(self.activations_hidden[i])*hidden_error_temp

		#Update both sets of weights:
		#output weights:
		for i in np.arange(self.N_hidden):
			for j in np.arange(self.N_outputs):
				self.weights2[j][i] += learning_rate*output_error[j]*self.activations_hidden[i]
		#input weights:
		for i in np.arange(self.N_inputs):
			for j in np.arange(self.N_hidden):
				self.weights1[j][i] += learning_rate*hidden_deltas[j]*self.activations_input[i]

		#Error
		#for j in np.arange(length):
		#		classifieds = np.zeros(length)
		#		classifieds[j] = self.Classify(vectors[:,j])
		#for i in np.arange(length):
		#	Error += 0.5*(targets[i] - classifieds[i])**2
		#print Error

		return Error


	def ClassifyPoint(self, Zoutput_temp):
		if Zoutput_temp[0] > 0.5:
			Zoutput = 1
		else:
			Zoutput = 0

		return Zoutput


	def Contour(self, initial, final, length):
		#Outputs points for a contour.
		vector_values = np.linspace(initial, final, length)
		x = y = vector_values
		z = np.zeros((length, length))

		for i in np.arange(length):
			for j in np.arange(length):
				z[i][j] = self.ClassifyPoint(self.Classify(np.array([x[i],y[j]])))

		return x,y,z


backprop_data = pickle.load(open("backprop-data.pkl", 'rb')) #Loads a dictionary of the perceptron data
labels = backprop_data['labels']
vectors = backprop_data['vectors']
network = Network(2, 10, 1)

#--------------------------------------------------------------------------------------------------
#Plotting a graph of the data points and contours before learning.
#--------------------------------------------------------------------------------------------------
#Plotting the points:
ones = [[],[]]   #Stores [[x's],[y's]] vectors with label=1
zeros = [[],[]]  #Stores [[x's],[y's]] vectors with label=0

for j in np.arange(len(labels)):
	if labels[j] == 1:
		ones[0].append(vectors[0][j])
		ones[1].append(vectors[1][j])
	else:
		zeros[0].append(vectors[0][j])
		zeros[1].append(vectors[1][j])


mpl.figure(1)
mpl.plot(ones[0], ones[1], 'o', label='1')
mpl.plot(zeros[0], zeros[1], 'o', label='0')
mpl.legend()

#Now for the contour plot:
X, Y, Z = network.Contour(-8, 8, 50)
mpl.contourf(X,Y,Z, levels=[0.0,1.0], extend='both')


#--------------------------------------------------------------------------------------------------
#Let's learn a thing or two... and plot it, of course.
#--------------------------------------------------------------------------------------------------
#print network.weights1
#print network.weights2
network.learn(vectors, labels, 100)
print network.weights1
print network.weights2
#print network.weights1
#print network.weights2

mpl.figure(2)
mpl.plot(ones[0], ones[1], 'o', label='1')
mpl.plot(zeros[0], zeros[1], 'o', label='0')
mpl.legend()

X, Y, Z = network.Contour(-8, 8, 50)
mpl.contourf(X,Y,Z, levels=[0.0,1.0], extend='both')
mpl.show()