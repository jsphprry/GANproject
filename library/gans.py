# libraries
import matplotlib.pyplot as plt
import numpy as np

import networks
import handling
import plotting




# test a grid of gan model configurations
# title       : title for the figures
# gan         : (latent space, generator, discriminator)
# datasets    : datasets and dataset metadata as (batched_data, plot_height, plot_width, plot_scale, data_image, data_title)
# param_space : parameter configurations as (number of training steps, generator learning rate, discriminator learning rate)
# save_plots  : flag for saving plots to file
# save_igis   : flag for saving intermediates to file
def gridTestModel(title, gan, param_space, datasets, save_plots, save_igis):
		
	# for each parameter configuration
	for tp in param_space:
	
		# for each dataset
		for ds in datasets:
			
			# test model
			testModel(title, gan, ds, tp, save_plots, save_igis)
			
			# reset model
			ldim, g, d = gan
			networks.resetNetwork(g)
			networks.resetNetwork(d)
			gan = ldim, g, d




# test a gan model
# title           : title for the figures
# gan             : (latent space, generator, discriminator)
# full_data       : batched dataset and dataset metadata as (batched_data, plot_height, plot_width, plot_scale, data_image, data_title)
# training_params : (#iterations, generator learning rate, discriminator learning rate)
# save_plots      : flag for saving plots to file
# save_igis       : flag for saving intermediates to file
def testModel(title, gan, full_data, training_params, save_plots, save_igis):
	
	# unpacking
	ldim, g, d = gan
	batched_data, plot_height, plot_width, plot_scale, data_image, data_title = full_data
	n_steps, g_eta, d_eta = training_params
	
	# train model
	g_costs, g_activs, d_costs, d_activs, igis = trainGAN(gan, training_params, batched_data)
	
	# set plotting SNS flag
	plotting.setSNS(save_plots)
	
	# plot dataset figure
	plotting.plotChannels(f"{data_title} training data", data_image, plot_height, plot_width, plot_scale)
	
	# plot training figure
	training_title = f"{title} {data_title} ({n_steps} {g_eta} {d_eta}) training figure"
	plotting.ganTrainingGraph(training_title, range(n_steps), g_costs, g_activs, d_costs, d_activs)
	
	# plot generated figure
	generated_title = f"{title} {data_title} ({n_steps} {g_eta} {d_eta}) generated figure"
	generated_images = [networks.noiseToImage(ldim,g)[0] for i in range(10)]
	plotting.plotChannels(generated_title, generated_images, 2, 5, 2)
	
	# plot progress figure
	progression_steps = 10
	progression_title = f"{title} {data_title} ({n_steps} {g_eta} {d_eta}) progression figure"
	progression_images = [igis[i] for i in range(0, n_steps, int(n_steps/progression_steps))]
	plotting.plotChannels(progression_title, progression_images, 1, progression_steps, 2)
	
	# save intermediate generated images
	if save_igis == True:
		handling.saveImages(igis, "igis/", f"saving {title} intermediate generated images...", "done.\n")




# train gan on batched data
# gan             : (latent space, generator, discriminator)
# training_params : (#iterations, generator learning rate, discriminator learning rate)
# batched_data    : batched dataset
def trainGAN(gan, training_params, batched_data):
	
	# unpacking
	ldim, g, d = gan
	n_steps, g_eta, d_eta = training_params
	
	# setup records
	g_costs = []
	g_activs = []
	d_costs = []
	d_activs = []
	igis = []
	
	# stochastic gradient descent
	for i in range(n_steps):
		
		# train step
		d_results, g_results = optimize(ldim, g, d, g_eta, d_eta, batched_data[i % len(batched_data)])
		
		# unpack results
		d_mean_cost, d_mean_activation = d_results
		g_mean_cost, g_mean_activation = g_results
		
		# record results
		d_costs.append(d_mean_cost)
		d_activs.append(d_mean_activation)
		g_costs.append(g_mean_cost)
		g_activs.append(g_mean_activation)
		
		# record generated images
		igis.extend([c for c in networks.noiseToImage(ldim,g)])
		
		print(f"step={i}, g_mean_cost={g_mean_cost:.2f}, g_mean_activation={g_mean_activation:.2f}")
		print(f"step={i}, d_mean_cost={d_mean_cost:.2f}, d_mean_activation={d_mean_activation:.2f}")
	
	# return records
	return (g_costs, g_activs, d_costs, d_activs, igis)




# optimize gan for one step
# ldim  : latent space dimensions, generator, discriminator)
# g     : generator network 
# d     : discriminator network
# g_eta : generator learning rate
# d_eta : discriminator learning rate
# data  : optization target
def optimize(ldim, g, d, g_eta, d_eta, data):
	
	# setup data
	true_false_data = []
	noise_data = []
	for x in data:
		true_false_data.append((x, 1))
		true_false_data.append((networks.noiseToImage(ldim, g), 0))
		noise_data.append((networks.noise(ldim), 1))
		noise_data.append((networks.noise(ldim), 1)) # ?
	
	# train steps
	# could be replaced with network.train* functions for multistep training
	# but this might require changes to the batching scheme
	d_results = networks.optimize(true_false_data, d, d_eta)
	g_results = networks.optimizeChained(noise_data, g, d, g_eta)
	
	# return results
	return (d_results, g_results)



