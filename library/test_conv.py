# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from handling import digits




# constants
x = np.pad(digits[2][0],4)
dk = np.array([
	[-1,-1,10],
	[-1,10,-1],
	[10,-1,-1],
	])
vk = np.array([
	[-1,10,-1],
	[-1,10,-1],
	[-1,10,-1],
	])
hk = np.array([
	[-1,-1,-1],
	[10,10,10],
	[-1,-1,-1],
	])
dk55 = np.array([
	[-1,-1,-1,-1,10],
	[-1,-1,-1,10,-1],
	[-1,-1,10,-1,-1],
	[-1,10,-1,-1,-1],
	[10,-1,-1,-1,-1],
	])
vk55 = np.array([
	[-1,-1,10,-1,-1],
	[-1,-1,10,-1,-1],
	[-1,-1,10,-1,-1],
	[-1,-1,10,-1,-1],
	[-1,-1,10,-1,-1],
	])
hk55 = np.array([
	[-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1],
	[10,10,10,10,10],
	[-1,-1,-1,-1,-1],
	[-1,-1,-1,-1,-1],
	])




# plotting functions for the convolution kernels used in this module
def kernels3x3():
	fig, axis = plt.subplots(1,3, figsize=(3,1))
	
	axis[0].imshow(dk)
	axis[1].imshow(vk)
	axis[2].imshow(hk)
	
	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)
	
	plt.tight_layout()
	plt.show()


def kernels5x5():
	fig, axis = plt.subplots(1,3, figsize=(3,1))

	axis[0].imshow(dk55)
	axis[1].imshow(vk55)
	axis[2].imshow(hk55)

	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




# valid convolutions of the convolution kernels with 'digits:2'
def validconv3x3():
	y_dk = signal.convolve2d(x, dk, 'valid')
	y_vk = signal.convolve2d(x, vk, 'valid')
	y_hk = signal.convolve2d(x, hk, 'valid')

	fig, axis = plt.subplots(1,4, figsize=(12,3))

	axis[0].imshow(x)
	axis[1].imshow(y_dk)
	axis[2].imshow(y_vk)
	axis[3].imshow(y_hk)

	axis[0].set_title("Input image")
	axis[1].set_title("Diagonal feature map")
	axis[2].set_title("Vertical feature map")
	axis[3].set_title("Horizontal feature map")

	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




def validconv5x5():
	y_dk55 = signal.convolve2d(x, dk55, 'valid')
	y_vk55 = signal.convolve2d(x, vk55, 'valid')
	y_hk55 = signal.convolve2d(x, hk55, 'valid')

	fig, axis = plt.subplots(1,4, figsize=(12,3))

	axis[0].imshow(x)
	axis[1].imshow(y_dk55)
	axis[2].imshow(y_vk55)
	axis[3].imshow(y_hk55)

	axis[0].set_title("Input image")
	axis[1].set_title("Diagonal feature map")
	axis[2].set_title("Vertical feature map")
	axis[3].set_title("Horizontal feature map")

	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




# comparison of valid and full convolution 
# with constant convolution kernel and input
def validfullconv3x3():
	y_dkv = signal.convolve2d(x, dk, 'valid')
	y_dkf = signal.convolve2d(x, dk, 'full')

	fig, axis = plt.subplots(1,3, figsize=(9,3))

	axis[0].imshow(x)
	axis[1].imshow(y_dkv)
	axis[2].imshow(y_dkf)

	axis[0].set_title("Input image")
	axis[1].set_title("Valid convolutional mapping")
	axis[2].set_title("Full convolutional mapping")

	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




# plots of the feature maps of each digit in digits
def alldigitsvalidconv3x3():
	fig, axis = plt.subplots(10,4, figsize=(4,10))

	# axis[0][0].set_title("Input image")
	# axis[0][1].set_title("Diagonal feature map")
	# axis[0][2].set_title("Vertical feature map")
	# axis[0][3].set_title("Horizontal feature map")

	for i, row in enumerate(axis):
		row[0].imshow(digits[i][0])
		row[1].imshow(signal.convolve2d(digits[i][0], dk, 'valid'))
		row[2].imshow(signal.convolve2d(digits[i][0], vk, 'valid'))
		row[3].imshow(signal.convolve2d(digits[i][0], hk, 'valid'))
		
		for a in row:
			a.xaxis.set_visible(False)
			a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




# main program
kernels3x3()
validconv3x3()
kernels5x5()
validconv5x5()
validfullconv3x3()
alldigitsvalidconv3x3()
