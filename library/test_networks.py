# libraries
import matplotlib.pyplot as plt
import numpy as np

import networks
import layers
import activations
import handling
import plotting




# training network with gradient descent
def gradientDescent(n_steps, learning_rate):
	
	# network
	input_shape = (1, 10, 10)
	n = [
		layers.Convolutional2D(input_shape, 3, 2),
		activations.Sigmoid(),
		layers.Reshape((2, 8, 8), (128, 1)),
		layers.Dense(128, 1),
		activations.Sigmoid()]
	
	# batched random target data
	target_activation = 1.0
	data = [[(networks.noise(input_shape), target_activation)]]
	
	# train network
	costs, activs = networks.trainNetwork(n_steps, data, n, learning_rate)
	
	# plot results
	plotting.costactivGraph(f"Optimizing network with target activation {target_activation}", range(n_steps), costs, activs)




# training network a in a->b with gradient descent
def chainedGradientDescent(n_steps, learning_rate):
	
	# network
	input_shape = (10,1)
	a = [
		layers.Dense(10,200),
		activations.Sigmoid(),
		layers.Dense(200,100),
		activations.Sigmoid(),
		layers.Reshape((100,1), (1,10,10))]
	b = [
		layers.Reshape((1,10,10), (100,1)),
		layers.Dense(100,200),
		activations.Sigmoid(),
		layers.Dense(200,1),
		activations.Sigmoid()]
	
	# batched random target data
	target_activation = 1.0
	data = [[(networks.noise(input_shape), target_activation)]]
	
	# train network
	costs, activs = networks.trainChainedNetwork(n_steps, data, a, b, learning_rate)
	
	# plot records
	plotting.costactivGraph(f"Optimizing chained network with target activation {target_activation}", range(n_steps), costs, activs)




# test ability to classify labeled digits
def fit_to_digits(n_steps, learning_rate):
	
	# network
	n = [
		layers.Dense(100,10),
		activations.Sigmoid()]
	
	target_x = [x.reshape((100,1)) for x in handling.digits]
	target_y = [handling.classLabel(i,10) for i in range(10)]
	target = list(zip(target_x,target_y))
	
	# train network
	costs, _ = networks.trainNetwork(n_steps, [target], n, learning_rate)
	
	# test performance
	print("calculating accuracy scores...")
	train_accuracy = networks.classificationAccuracy(target, n)
	print(f"train set accuracy score={train_accuracy:.2f}%")
	
	# plot results
	plotting.costGraph("Dense network labelled digits optimization", range(n_steps), costs)




# test ability to classify MNIST dataset
def fit_to_MNIST(epochs, n_steps, learning_rate):
	
	# network
	n = [
		layers.Dense(784,200),
		activations.Sigmoid(),
		layers.Dense(200,10),
		activations.Sigmoid()]
	
	# partition mnist into testing data and training data
	data = handling.loadMNIST("../data/mnist/", (784,1)) # 60_000 items
	partition = int(0.2* len(data))
	
	testing_data = data[:partition]  # 12_000 items
	training_data = data[partition:] # 48_000 items
	
	# train network on batched data for a number of epochs
	costs = []
	for e in range(epochs):
		batched_data = handling.shuffleSplit(training_data, n_steps)
		epoch_costs, _ = networks.trainNetwork(n_steps, batched_data, n, learning_rate)
		costs.extend(epoch_costs)
	
	# test performance
	print("calculating accuracy scores...")
	test_accuracy = networks.classificationAccuracy(testing_data, n)
	train_accuracy = networks.classificationAccuracy(training_data, n)
	print(f"test set accuracy score={test_accuracy:.2f}%")
	print(f"train set accuracy score={train_accuracy:.2f}%")
	
	# plot results
	plotting.costGraph("Dense network MNIST optimization", range(epochs*n_steps), costs)




# calculate cross validation accuracy scores for network with mnist
def CV_MNIST(data_folds, n_epochs, n_steps, learning_rate):
	
	# network
	n = [
		layers.Dense(784,200),
		activations.Sigmoid(),
		layers.Dense(200,10),
		activations.Sigmoid()]
	
	# calculate network cross validation accuracy scores for mnist data
	data = handling.loadMNIST("../data/mnist/", (784,1))[:20_000]
	cv_test, cv_train = networks.crossValidate(data, data_folds, n_epochs, n_steps, n, learning_rate)




# optimization
gradientDescent(100, 0.01)
chainedGradientDescent(100, 0.1)

# classification
fit_to_digits(100, 1)
fit_to_MNIST(5, 1000, 0.1)
CV_MNIST(10, 5, 1000, 0.1)
