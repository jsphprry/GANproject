# libraries
import matplotlib.pyplot as plt
import numpy as np

import networks
import layers
import activations
import handling
import plotting




# two dimensional point generator
def pointGAN(n_steps, g_eta, d_eta):
	
	ldim = (1,1)
	g = [
		layers.Dense(1,2),
		activations.Sigmoid(),
		layers.Dense(2,2),
		activations.Sigmoid()]
	d = [
		layers.Dense(2,2),
		activations.Sigmoid(),
		layers.Dense(2,1),
		activations.Sigmoid()]
	
	points_data = [np.array([[0.5],[0.5]])]
	
	# setup records
	g_costs = []
	g_activs = []
	d_costs = []
	d_activs = []
	igis = []
	
	# gd
	for i in range(n_steps):
		
		# train step
		g_results, d_results = networks.ganGradientStep(points_data, ldim,g,d, g_eta, d_eta)
		
		# unpack results
		g_mean_cost, g_mean_activation = g_results
		d_mean_cost, d_mean_activation = d_results
		
		# record results
		g_costs.append(g_mean_cost)
		g_activs.append(g_mean_activation)
		d_costs.append(d_mean_cost)
		d_activs.append(d_mean_activation)
		
		# record generated point
		igis.append(networks.noiseToImage(ldim,g).flatten())
		
		print(f"step={i}")
		print(f"d_mean_cost={d_mean_cost:.2f} d_mean_activation={d_mean_activation:.2f}")
		print(f"g_mean_cost={g_mean_cost:.2f} g_mean_activation={g_mean_activation:.2f}")
	
	plotting.ganTrainingGraph("points data training", range(n_steps),g_costs,g_activs,d_costs,d_activs,show=True)
	
	generated_x = [a for a,b in igis]
	generated_y = [b for a,b in igis]
	generated_c = [c/len(igis) for c in reversed(range(len(igis)))]
	
	data_x = [a for a,b in points_data]
	data_y = [b for a,b in points_data]
	
	final_gen = [networks.noiseToImage(ldim,g) for i in range(10)]
	final_gen_x = [a for a,b in final_gen]
	final_gen_y = [b for a,b in final_gen]
	
	plt.scatter(generated_x, generated_y, c=generated_c, cmap='gray')
	plt.axhline(y=0.5, color='r', linestyle='-')
	plt.axvline(x=0.5, color='r', linestyle='-')
	plt.scatter(data_x, data_y, label="Target", c='red')
	plt.scatter(final_gen_x, final_gen_y, label="Generated", c='green')
	plt.xlabel("x value")
	plt.ylabel("y value")
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()




# main program
pointGAN(10_000,1,1)
