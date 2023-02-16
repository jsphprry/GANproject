# libraries
import matplotlib.pyplot as plt
import numpy as np

import gans
import layers
import activations
import handling





# layers.Dense GAN image generator
# datasets    : datasets and dataset metadata as a list of (batched_data, plot_height, plot_width, plot_scale, data_image, data_title)
# param_space : parameter configurations as a list of (number of training steps, generator learning rate, discriminator learning rate)
# save_plots  : flag for saving plots to file
# save_igis   : flag for saving intermediates to file
def denseGAN(datasets, param_space, save_plots, save_igis):
	
	# the model discussed in the report
	"""
	# GAN
	# latent space
	ldim = (10,1)
	
	# generator network
	g = [
		layers.Dense(10,200),
		activations.Sigmoid(),
		layers.Dense(200,200),
		activations.Sigmoid(),
		layers.Dense(200,100),
		activations.Sigmoid(),
		layers.Dense(100,100),
		activations.Sigmoid(),
		layers.Reshape((100,1), (1,10,10))]
	
	# discriminator network
	d = [
		layers.Reshape((1,10,10), (100,1)),
		layers.Dense(100,200),
		activations.Sigmoid(),
		layers.Dense(200,200),
		activations.Sigmoid(),
		layers.Dense(200,200),
		activations.Sigmoid(),
		layers.Dense(200,1),
		activations.Sigmoid()]
	"""
	
	# this model performs better on the same tasks because it is simpler
	
	# GAN
	# latent space
	ldim = (10,1)
	
	# generator network
	g = [
		layers.Dense(10,200),
		activations.Sigmoid(),
		layers.Dense(200,200),
		activations.Sigmoid(),
		layers.Dense(200,100),
		activations.Sigmoid(),
		layers.Reshape((100,1), (1,10,10))]
	
	# discriminator network
	d = [
		layers.Reshape((1,10,10), (100,1)),
		layers.Dense(100,200),
		activations.Sigmoid(),
		layers.Dense(200,200),
		activations.Sigmoid(),
		layers.Dense(200,1),
		activations.Sigmoid()]
			
	# test model
	gans.gridTestModel("denseGAN", (ldim,g,d), datasets, param_space, save_plots, save_igis)




# Convolutional GAN image generator
# datasets    : datasets and dataset metadata as a list of (batched_data, plot_height, plot_width, plot_scale, data_image, data_title)
# param_space : parameter configurations as a list of (number of training steps, generator learning rate, discriminator learning rate)
# save_plots  : flag for saving plots to file
# save_igis   : flag for saving intermediates to file
def convGAN(datasets, param_space, save_plots, save_igis):
	
	# GAN
	# latent space
	ldim = (1, 4, 4)
	
	# generator network
	g = [
		layers.ConvolutionalTranspose2D(ldim, 3, 5),
		activations.Sigmoid(),
		layers.ConvolutionalTranspose2D((5, 6, 6), 3, 5),
		activations.Sigmoid(),
		layers.ConvolutionalTranspose2D((5, 8, 8), 3, 1),
		activations.Sigmoid()]
	
	# discriminator network
	l0_features = 10
	ln_dense_ns = 200
	d = [
		layers.Convolutional2D((1, 10, 10), 3, l0_features),
		layers.Reshape((l0_features, 8, 8), (l0_features*64, 1)),
		activations.Sigmoid(),
		layers.Dense(l0_features*64, ln_dense_ns),
		activations.Sigmoid(),
		layers.Dense(ln_dense_ns, 1),
		activations.Sigmoid()]
	
	# test model
	gans.gridTestModel("convGAN", (ldim,g,d), datasets, param_space, save_plots, save_igis)
	




# parameter configurations
param_space = [
	(1000,0.5,0.5),
	(1000,0.15,0.15),
	(1000,0.05,0.05),
	(1000,0.01,0.01)]

# targets
datasets = [
	([[handling.true_image]], 1, 1, 3, handling.true_image, "true_image"),
	([handling.digits], 2, 5, 2, [x[0] for x in handling.digits], "digits")]

# main program
denseGAN(param_space, datasets, False, False)
convGAN(param_space, datasets, False, False)
