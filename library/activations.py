# libraries
import numpy as np

# local libraries
from layers import Layer




# activation layer superclass
class Activation(Layer):
	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime
	
	def forward(self, x):
		self.input = x
		return self.activation(self.input)
	
	def backward(self, output_gradient):
		return np.multiply(output_gradient, self.activation_prime(self.input))




# leaky ReLU activation layer
class LeakyReLU(Activation):
	def __init__(self, a=0.01):
		self.a = a

		def leaky_relu(x):
			return np.where(x > 0, x, x * self.a)
		
		def leaky_relu_prime(x):
			return np.where(x > 0, 1, self.a)
		
		super().__init__(leaky_relu, leaky_relu_prime)




# ReLU activation layer
class ReLU(Activation):
	def __init__(self):
		def relu(x):
			return np.maximum(x, 0)
		
		def relu_prime(x):
			return np.where(x > 0, 1, 0)
		
		super().__init__(relu, relu_prime)




# Tanh activation layer
class Tanh(Activation):
	def __init__(self):
		def tanh(x):
			return np.tanh(x)
		
		def tanh_prime(x):
			return 1 - np.tanh(x) ** 2
		
		super().__init__(tanh, tanh_prime)




# Sigmoid activation layer
class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			return 1 / (1 + np.exp(-x))
		
		def sigmoid_prime(x):
			s = sigmoid(x)
			return s * (1 - s)
		
		super().__init__(sigmoid, sigmoid_prime)
