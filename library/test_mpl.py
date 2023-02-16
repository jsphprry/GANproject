# libraries
import matplotlib.pyplot as plt
import numpy as np

from layers import MaxPooling2D




# test function for MaxPooling2D layer
# prints matrix form to terminal and
# plot output with pyplot
# derivative should be 1 for max values 
# and 0 for non-max values
def test_mpl(s, p):
	l = MaxPooling2D((1,s,s), (p,p))
	x = np.arange(s*s) / (s*s)
	np.random.shuffle(x)
	x = x.reshape((1,4,4))

	print("\nInput\n",x)
	print("\nPool\n",l.forward(x))
	print("\nMask\n",l.input_mask)

	fig, axis = plt.subplots(1, 3, figsize=(9,3))
	axis[0].set_title("Input")
	axis[0].imshow(x[0], vmin=0, vmax=1)
	axis[1].set_title("Max Pool")
	axis[1].imshow(l.forward(x)[0], vmin=0, vmax=1)
	axis[2].set_title("Derivative")
	axis[2].imshow(l.input_mask[0], vmin=0, vmax=1)
	for a in axis:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	plt.tight_layout()
	plt.show()




# main program
test_mpl(4,2)
