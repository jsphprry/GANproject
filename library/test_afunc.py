# libraries
import matplotlib.pyplot as plt
import numpy as np

from activations import Sigmoid, Tanh, ReLU, LeakyReLU




# test program for the activation function layers.
# the plots for each activation should match the 
# expected plots for each activation function
def test_afunc(minx, maxx, res, a=0.1):
	
	sigmoid = Sigmoid()
	tanh = Tanh()
	relu = ReLU()
	lrelu = LeakyReLU(a=a)
	
	
	graph_x = np.linspace(minx, maxx, res)
	graph_y_sigmoid = []
	graph_y_sigmoid_delta = []
	graph_y_tanh = []
	graph_y_tanh_delta = []
	graph_y_relu = []
	graph_y_relu_delta = []
	graph_y_lrelu = []
	graph_y_lrelu_delta = []
	for x in graph_x:
		graph_y_sigmoid.append(sigmoid.forward(x))
		graph_y_sigmoid_delta.append(sigmoid.backward(1))
		graph_y_tanh.append(tanh.forward(x))
		graph_y_tanh_delta.append(tanh.backward(1))
		graph_y_relu.append(relu.forward(x))
		graph_y_relu_delta.append(relu.backward(1))
		graph_y_lrelu.append(lrelu.forward(x))
		graph_y_lrelu_delta.append(lrelu.backward(1))
	
	
	fig, axis = plt.subplots(1, 4, figsize=(12,3))
	axis[0].set_title("Sigmoid")
	axis[0].plot(graph_x, graph_y_sigmoid)
	axis[0].plot(graph_x, graph_y_sigmoid_delta)
	axis[0].grid()
	axis[1].set_title("Tanh")
	axis[1].plot(graph_x, graph_y_tanh)
	axis[1].plot(graph_x, graph_y_tanh_delta)
	axis[1].grid()
	axis[2].set_title("ReLU")
	axis[2].plot(graph_x, graph_y_relu)
	axis[2].plot(graph_x, graph_y_relu_delta)
	axis[2].grid()
	axis[3].set_title("Leaky ReLU")
	axis[3].plot(graph_x, graph_y_lrelu)
	axis[3].plot(graph_x, graph_y_lrelu_delta)
	axis[3].grid()
	plt.tight_layout()
	plt.show()




# main program
test_afunc(-5,5,100)
