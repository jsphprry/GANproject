# libraries
import numpy as np
import matplotlib.pyplot as plt

import costs
from layers import Layer
import handling




# classification accuracy score
# data :
# n    :
def classificationAccuracy(data, n):
	n_correct = 0
	data_len = len(data)
	
	for x, y in data:
		if forwardPropagate(x,n).argmax() == y.argmax():
			n_correct += 1
	
	return 100 * (n_correct / data_len)




# cross validation accuracy score
# todo: better return info, remove plotting and printing code 
#
#
#
#
#
#
def crossValidate(data, data_folds, n_epochs, n_steps, n, eta):
	
	# prepare variables
	partition = int(len(data) / data_folds)
	nd_data   = np.array(data, dtype=object)
	test_scores  = np.zeros(data_folds)
	train_scores = np.zeros(data_folds)
	
	# calculate classification accuracy scores for k-folds of data
	for k in range(data_folds):
		
		# reset network parameters
		resetNetwork(n)
		
		# partition data into test and train
		testing_data  = list(nd_data[:partition])
		training_data = list(nd_data[partition:])
		nd_data       = np.roll(nd_data, partition)
		
		# train network
		
		#####
		fold_costs = []
		#####
		
		for e in range(n_epochs):
			batched_data = handling.shuffleSplit(training_data, n_steps)
			
			#####
			costs, _ = trainNetwork(n_steps, batched_data, n, eta)
			fold_costs.extend(costs)
		plt.plot(range(n_steps*n_epochs), fold_costs)
		#####
			
		
		# calculate classification accuracy scores
		test_scores[k] = classificationAccuracy(testing_data, n)
		train_scores[k] = classificationAccuracy(training_data, n)
		print(f"fold={k}, test set accuracy score={test_scores[k]:.5f}%")
		print(f"fold={k}, train set accuracy score={train_scores[k]:.5f}%")
	
	# calculate mean classification accuracy scores
	mean_test_score = np.mean(test_scores)
	mean_train_score = np.mean(train_scores)
	print(f"Test CV score={mean_test_score:.2f}")
	print(f"Train CV score={mean_train_score:.2f}")
	
	#####
	plt.title("Dense network MNIST cross-validation")
	plt.ylabel("mean_cost")
	plt.xlabel("training step")
	plt.get_current_fig_manager().set_window_title("Dense network MNIST cross-validation")
	plt.tight_layout()
	plt.grid()
	plt.show()
	#####
	
	return (mean_test_score, mean_train_score)




# train network on batched data
#
#
#
#
def trainNetwork(n_steps, data, n, eta):
	
	# setup records
	costs = []
	activs = []
	
	# stochastic gradient descent
	for i in range(n_steps):
		
		# train step
		mean_cost, mean_activation = optimize(data[i % len(data)], n, eta)
		
		# record results
		costs.append(mean_cost)
		activs.append(mean_activation)
		print(f"step={i}, mean_cost={mean_cost:.2f}, mean_activation={mean_activation:.2f}")
	
	# return records
	return (costs, activs)


# train network a in a->b on batched data
#
#
#
#
#
def trainChainedNetwork(n_steps, data, a, b, eta):
	
	# setup records
	costs = []
	activs = []
	
	# stochastic gradient descent
	for i in range(n_steps):
		
		# train step
		mean_cost, mean_activation = optimizeChained(data[i % len(data)], a, b, eta)
		
		# record results
		costs.append(mean_cost)
		activs.append(mean_activation)
		print(f"step={i}, mean_cost={mean_cost:.2f}, mean_activation={mean_activation:.2f}")
	
	# return records
	return (costs, activs)




# update network a for one step
# data : optimization target
# a    : network
# eta  : learning rate
def optimize(data, a, eta):
	data_length = len(data)
	sum_cost = 0.0
	sum_activation = 0.0
	
	# sum gradients
	for x, y in data:
		
		# forward propagate x through n
		activation = forwardPropagate(x, a)
		
		# record results
		sum_cost += costs.crossEntropy(activation,y)
		sum_activation += np.mean(activation)
		
		# backward propagate error through n, keep gradient sum
		error = costs.crossEntropyDerivative(activation,y)
		delta = backwardPropagateSum(error,a)
	
	# apply average of gradient sum
	applyGradient(a,eta/data_length)
	
	# return statistics
	mean_cost = sum_cost/data_length
	mean_activation = sum_activation/data_length
	return (mean_cost, mean_activation)




# update network a in a->b for one step
# data : optimization target
# a    : optimizing network
# b    : static network
# eta  : learning rate
def optimizeChained(data, a, b, eta):
	data_length = len(data)
	sum_cost = 0.0
	sum_activation = 0.0
	
	# sum gradients
	for x, y in data:
		
		# forward propagate x through a through b
		activation = forwardPropagate(forwardPropagate(x,a),b)
		
		# record results
		sum_cost += costs.crossEntropy(activation,y)
		sum_activation += np.mean(activation)
		
		# backward propagate error through b discarding gradients, then through a summing gradients
		error = costs.crossEntropyDerivative(activation,y)
		delta = backwardPropagateSum(backwardPropagate(error, b, 0), a)
	
	# apply average of gradient sum to a
	applyGradient(a,eta/data_length)
	
	# return statistics
	mean_cost = sum_cost/data_length
	mean_activation = sum_activation/data_length
	return (mean_cost, mean_activation)




# array of random values
# shape : shape of array
def noise(shape):
	return np.random.randn(*shape)




# forward propagate noise through n
# input_shape : shape of input array
# n           : network
def noiseToImage(input_shape, n):
	return forwardPropagate(noise(input_shape),n)




# forward propagate x through n
# x : input column vector
# n : network
def forwardPropagate(x, n):
	activation = x
	for l in n:
		activation = l.forward(activation)
	
	return activation




# backward propagate delta through n, 
# apply gradient to trainables
# delta : derivative with respect to the outputs
# n     : network
# eta   : learning rate
def backwardPropagate(delta, n, eta):
	for l in reversed(n):
		delta = l.backward(delta)
		l.update(eta)
	
	return delta




# backward propagate delta through n, 
# keep a sum of gradients instead of 
# applying gradient to trainables
# delta : derivative with respect to the outputs
# n     : network
def backwardPropagateSum(delta, n):
	for l in reversed(n):
		delta = l.backward(delta)
	
	return delta




# apply gradient sum to trainables,
# gradient sum can be discarded by 
# passing 0 for eta
# n   : network
# eta : learning rate
def applyGradient(n, eta):
	for l in n:
		l.update(eta)




# reset network trainables
# n : network
def resetNetwork(n):
	for l in n:
		l.reset()

