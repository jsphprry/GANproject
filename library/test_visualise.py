# libraries
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D
import visualkeras
from PIL import ImageFont




# visualkeras representation of test_gans.denseGAN
def denseGAN():
	# setup network
	ldim = 10
	gdim = [ldim,200,200,100]
	ddim = [gdim[-1],200,200,1]
	
	model = Sequential()
	model.add(InputLayer(input_shape=ldim))
	
	for l in gdim[1:] + ddim:
		model.add(Dense(l))
	
	visualkeras.layered_view(model, to_file="../figures/visualise_denseGAN.png",
							legend=True, type_ignore=[Flatten, Reshape])




# visualkeras representation of test_gans.convGAN
def convGAN():
	model = Sequential()
	
	gan = [
		InputLayer(input_shape=(4,4,10)),
		Conv2DTranspose(10,3),
		Conv2DTranspose(10,3),
		Conv2DTranspose(10,3),
		Conv2D(10, 3),
		Flatten(),
		Dense(200),
		Dense(1)]
	
	for l in gan:
		model.add(l)
	
	visualkeras.layered_view(model, to_file="../figures/visualise_convGAN.png", 
							legend=True, type_ignore=[Flatten, Reshape])




# visualkeras representations of undocumented GANs
def altGAN(config_index):
	
	# bound config_index
	config_index = max(0, min(3, config_index))
	
	g1 = [
		InputLayer(input_shape=16), 
		Dense(160),
		Reshape((4,4,10)), 
		Conv2DTranspose(10,3), 
		Conv2DTranspose(10,3), 
		Conv2DTranspose(1,3)]
	 
	d1 = [
		Conv2D(5, 3), 
		Conv2D(5, 3), 
		Flatten(), 
		Dense(100), 
		Dense(1)]
	
	g2 = [
		InputLayer(input_shape=(1,4,4)),
		Conv2DTranspose(5,3),
		Conv2D(5,3),
		Conv2DTranspose(5,3),
		Conv2DTranspose(5,3),
		Conv2DTranspose(1,3)]
	
	d2 = [
		Conv2D(5,3),
		Conv2D(5,3),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),
		Dense(1)]

	configs = [
		(g1, d1),
		(g1, d2),
		(g2, d2),
		(g2, d1)]
	
	model = Sequential()
	g, d = configs[config_index]
	for l in g + d:
		model.add(l)
	
	visualkeras.layered_view(model, to_file=f"../figures/vis_altGAN{config_index}.png", 
							legend=True, type_ignore=[Flatten, Reshape])




# main program
denseGAN()
convGAN()
for i in range(4):
	altGAN(i)
