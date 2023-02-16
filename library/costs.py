# libraries
import numpy as np




# mean squared cost
def mse(a, y):
	return np.mean(np.power(y - a, 2))

def mseDerivative(a, y):
	return 2 * (a - y) / np.size(y)




# cross entropy cost
def crossEntropy(a, y):
	return np.mean(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def crossEntropyDerivative(a, y):
	return ( ((1-y) / (1-a)) - (y / a) ) / np.size(y)

