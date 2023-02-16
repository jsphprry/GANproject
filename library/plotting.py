# libraries
import matplotlib.pyplot as plt
import numpy as np

import handling


# flag to show or save figures
SAVE_NOT_SHOW = False

def setSNS(b):
	SAVE_NOT_SHOW = b


# plot graph of mean_cost against training step
def costGraph(title, x, y):
	plt.plot(x,y)
	plt.title(title)
	plt.ylabel("mean_cost")
	plt.xlabel("training step")
	
	plt.get_current_fig_manager().set_window_title(title)
	plt.tight_layout()
	plt.grid()
	plt.show()


# plot mean_cost and mean_activation against training step
def costactivGraph(title, x, y_costs, y_activs):
	fig, axis = plt.subplots(1, 2, figsize=(12,4))
	fig.suptitle(title)
	
	axis[0].plot(x,y_costs)
	axis[0].set_ylabel("mean_cost")
	axis[0].set_xlabel("training step")
	
	axis[1].plot(x,y_activs)
	axis[1].set_ylabel("mean_activation")
	axis[1].set_xlabel("training step")
	axis[1].set_ylim(0,1)
	
	axis[0].grid()
	axis[1].grid()
	
	plt.get_current_fig_manager().set_window_title(title)
	plt.show()


# plot graph of gan training results
def ganTrainingGraph(title, graph_x, graph_y_g_costs, graph_y_g_activs, graph_y_d_costs, graph_y_d_activs):
	
	fig, axis = plt.subplots(2, 2, figsize=(12,6))
	
	axis[0][0].plot(graph_x, graph_y_g_costs)
	axis[0][0].set_title("Generator")
	axis[0][0].set_ylabel("mean_cost")
	axis[0][0].grid()
	
	axis[0][1].plot(graph_x, graph_y_d_costs)
	axis[0][1].set_title("Discriminator")
	axis[0][1].set_ylabel("mean_cost")
	axis[0][1].grid()
	
	axis[1][0].plot(graph_x, graph_y_g_activs)
	axis[1][0].set_ylabel("mean_activation")
	axis[1][0].set_xlabel("training step")
	axis[1][0].set_ylim(0,1)
	axis[1][0].grid()
	
	axis[1][1].plot(graph_x, graph_y_d_activs)
	axis[1][1].set_ylabel("mean_activation")
	axis[1][1].set_xlabel("training step")
	axis[1][1].set_ylim(0,1)
	axis[1][1].grid()
	
	if SAVE_NOT_SHOW:
		plt.savefig(f"../figures/{title}.png", format="png")
		plt.clf()
	else:
		plt.show()


# plot image channels
def plotChannels(title, image, height, width, scale, show_axis=False):
	
	fig, axis = plt.subplots(height, width, figsize=(width*scale,height*scale), squeeze=False)
	
	for i in range(height):
		for j in range(width):
			axis[i][j].imshow(image[(i * width) + j])
			axis[i][j].xaxis.set_visible(show_axis)
			axis[i][j].yaxis.set_visible(show_axis)
	
	plt.get_current_fig_manager().set_window_title(title)
	plt.tight_layout()
	
	if SAVE_NOT_SHOW:
		plt.savefig(f"../figures/{title}.png", format="png")
		plt.clf()
	else:
		plt.show()
	

# plot true_image
def plot_true_image():
	fig = plt.imshow(handling.true_image[0])
	plt.colorbar()
	plt.get_current_fig_manager().set_window_title("true_image")
	plt.show()


# plot digits set
def plot_digits():
	plotChannels("digits", [x[0] for x in handling.digits], 2, 5, 2)
