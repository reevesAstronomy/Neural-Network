#--------------------------------------------------------------------------------------------------
#Computational Neuroscience Assignment: Regression, Classification, and Neural Networks
#Question 3: Feedforward network trained using gradient descent and backpropagation.
#BIOL 487 / SYDE 552
#Andrew Reeves - 20459205
#--------------------------------------------------------------------------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as mpl

#--------------------------------------------------------------------------------------------------
def SortPoints(input_vectors, labels):
	#Sort the points, whether they are labelled '1' or '0':
	ones = [[],[]]   #Stores [[x's],[y's]] vectors with label=1
	zeros = [[],[]]  #Stores [[x's],[y's]] vectors with label=0

	for j in np.arange(len(labels)):
		if labels[j] == 1:
			ones[0].append(input_vectors[0][j])
			ones[1].append(input_vectors[1][j])
		else:
			zeros[0].append(input_vectors[0][j])
			zeros[1].append(input_vectors[1][j])

	return zeros, ones

#--------------------------------------------------------------------------------------------------
#Activation functions:
def hid_act(a): return np.tanh(a)
def out_act(a): return (1.0/(1.0 + np.exp(-a)))

#Derivatives of the activation functions:
def d_hid_act(z): return (1.0 - z**2)
def d_out_act(z): return z*(1.0 - z)

#--------------------------------------------------------------------------------------------------
class Network:
	def __init__(self, input_vectors, labels, N_inputs=2, N_hidden=10, N_outputs=1):
		#Initialize the neural network. Note: "N" stands for "Number of _".
		self.N_in = N_inputs + 1 #Includes a bias weight vector.
		self.N_hid = N_hidden
		self.N_out = N_outputs

		#Include the training data:
		self.vectors = input_vectors
		self.labels = labels

		#Initialize the weights with random values.
		self.weightsIn = np.random.rand(self.N_in, self.N_hid)
		self.weightsOut = np.random.rand(self.N_hid, self.N_out)

		#"acts" = "activations", "in" = "inputs", "hid" = "hidden", "out" = "outputs"
		self.acts_in = np.zeros(self.N_in) + 1
		self.acts_hid = np.zeros(self.N_hid) + 1
		self.acts_out = np.zeros(self.N_out) + 1


	def Classify(self, inputs):
		#Given an input, what does the network output?

		#Load inputs into the input activations, being careful to keep the bias input node's activation at zero.
		for i in np.arange(len(inputs)):
			self.acts_in[i] = inputs[i]
		
		#Calculate the hidden node activations and then the output activations using numpy matrix multiplication:
		self.acts_hid = hid_act(np.dot(np.transpose(self.weightsIn), self.acts_in))
		self.acts_out = out_act(np.dot(np.transpose(self.weightsOut), self.acts_hid))

		return self.acts_out #Output of network


	def Backprop(self, targets, learningrate):
		#Use backpropagation to update the weight values.

		#Calculate the deltas:
		output_deltas = (targets - self.acts_out) * d_out_act(self.acts_out)
		hidden_deltas = d_hid_act(self.acts_hid) * np.dot(self.weightsOut, output_deltas)
		
		#Update the output weights:
		shift = np.outer(self.acts_hid, output_deltas)
		self.weightsOut += learningrate * shift

		#Update the input weights:
		shift = np.outer(self.acts_in, hidden_deltas)
		self.weightsIn += learningrate * shift


	def Contour(self):
		#Run through the points in a grid and classify the points. The classifications given for each
		#(X,Y) coordinate in the grid can then be used to create a contour of the network output for
		#inputs over the range [-8,8].
		x = y = np.linspace(-8,8,50)
		Z = np.zeros([len(x), len(y)])

		for i in range(len(x)):
		    for j in range(len(y)):
		        Z[i][j] = float(self.Classify([x[i], y[j]])[0])>0.50

		X, Y = np.meshgrid(x,y)

		return X, Y, Z


	def Learn(self, iterations, learningrate=0.5):
		#Iterate through all of the training data "iterations" number of times. After each point
		#is classified, the output is used for backpropagation to update the weights of the network.

		for i in np.arange(iterations):
			for pos in np.arange(len(self.labels)): #pos is the index of the training data point being classified.
				self.Classify(self.vectors[:,pos])
				self.Backprop(self.labels[pos],learningrate)


#--------------------------------------------------------------------------------------------------
#Load the data from the pickle file and get it ready to rock-and-roll:
training_data = pickle.load(open('backprop-data.pkl','rb'))
input_vectors = training_data['vectors'] #Training inputs
labels = training_data['labels'] #Training labels
zeros, ones = SortPoints(input_vectors, labels) #Point positions for the ones or zeros (for plotting).

#Create the network:
network = Network(input_vectors, labels) #Creating the network requires training data to be input!

#--------------------------------------------------------------------------------------------------
#Contour plot before training the network:
mpl.figure(1)
X, Y, Z = network.Contour()
mpl.contourf(X,Y,Z, levels=[0.0,1.0], extend='both')
mpl.plot(zeros[0], zeros[1], 'o', label='0')
mpl.plot(ones[0], ones[1], 'o', label='1')

mpl.xlim(-8,8)
mpl.ylim(-8,8)
mpl.xlabel('x')
mpl.ylabel('y')
mpl.legend()
mpl.title('Question 3: Before Training')
mpl.savefig('A1Q3 - Before Training')

#--------------------------------------------------------------------------------------------------
#Train the network all the things:
network.Learn(1000)

#--------------------------------------------------------------------------------------------------
#After training contour plot:
mpl.figure(2)
X, Y, Z = network.Contour()
mpl.contourf(X,Y,Z, levels=[0.0,1.0], extend='both')
mpl.plot(zeros[0], zeros[1], 'o', label='0')
mpl.plot(ones[0], ones[1], 'o', label='1')

mpl.xlim(-8,8)
mpl.ylim(-8,8)
mpl.xlabel('x')
mpl.ylabel('y')
mpl.legend()
mpl.title('Question 3: After Training')
mpl.savefig('A1Q3 - After Training')
mpl.show()