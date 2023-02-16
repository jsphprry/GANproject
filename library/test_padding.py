# libraries
import matplotlib.pyplot as plt
import numpy as np

from layers import ZeroPadding2D




# print input processed with forward and backwards methods
# forward method should show appropriate padding to pixels and edges
# backward method should contain only the inputs from the corresponding 
# original pixel values in the forward method
def test_padding(d,h,w,ep,ip):
	
	input_shape = (d,h,w)
	output_shape = (d, (h*(1+ip) + 2*ep + ip), (w*(1+ip) + 2*ep + ip))
	
	input_volume = d*h*w
	output_volume = d * (h*(1+ip) + 2*ep + ip) * (w*(1+ip) + 2*ep + ip)
	
	
	l = ZeroPadding2D(input_shape, ep, ip)
	fx = np.arange(input_volume).reshape(input_shape) + 1
	bx = np.arange(output_volume).reshape(output_shape) + 1
	fy = l.forward(fx)
	by = l.backward(bx)
	
	print("ip kernel")
	print(l.ip_kernel)
	print("forward x")
	print(fx)
	print("forward y")
	print(fy)
	print("backward x")
	print(bx)
	print("backward y")
	print(by)




# visualise with pyplot
def test_padding_extra_intra(d,h,w,ep,ip):
	
	fx = np.arange(h*w).reshape((1,h,w)) + 1
	l0 = ZeroPadding2D((1,h,w), ep, 0)
	l1 = ZeroPadding2D((1,h,w), 0, ip)
	
	fig, axis = plt.subplots(1, 3, figsize=(9,3))
	axis[0].set_title("Input")
	axis[0].imshow(fx[0])
	axis[1].set_title("Padded edges")
	axis[1].imshow(l0.forward(fx)[0])
	axis[2].set_title("Padded pixels")
	axis[2].imshow(l1.forward(fx)[0])
	plt.tight_layout()
	plt.show()




# terminal program
test_padding(1,4,4,1,1)

# graphical program
test_padding_extra_intra(1,4,4,1,1)
