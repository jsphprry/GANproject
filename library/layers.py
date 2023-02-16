# libraries
import numpy as np
from scipy import signal




# scalar for RNG values
TRAINABLE_INIT_SCALE = 0.25

def setTIS(n):
	TRAINABLE_INIT_SCALE = n




# layer superclass
class Layer:
	def __init__(self):
		self.input = None
		self.output = None
	
	# forward propogation function
	def forward(self, x):
		return x
	
	# backward propogation function
	def backward(self, output_delta):
		return output_delta
	
	# update trainable parameters
	def update(self, eta):
		pass
	
	# reset trainable parameters
	def reset(self):
		pass




# fully connected layer
class Dense(Layer):
	def __init__(self, input_size, output_size):
		super().__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		
		self.biases = np.random.randn(output_size, 1) * TRAINABLE_INIT_SCALE
		self.weights = np.random.randn(output_size, input_size) * TRAINABLE_INIT_SCALE
		
		self.biases_grads_sum = np.zeros((output_size, 1))
		self.weights_grads_sum = np.zeros((self.output_size, self.input_size))
	
	def forward(self, x):
		self.input = x
		self.output = np.dot(self.weights, self.input) + self.biases
		return self.output
	
	def backward(self, output_delta):
		biases_delta = output_delta
		weights_delta = np.dot(output_delta, self.input.T)
		input_delta = np.dot(self.weights.T, output_delta)
		
		self.biases_grads_sum += biases_delta
		self.weights_grads_sum += weights_delta
		return input_delta
	
	def update(self, eta):
		self.biases -= self.biases_grads_sum * eta
		self.weights -= self.weights_grads_sum * eta
		
		self.biases_grads_sum = np.zeros((self.output_size, 1))
		self.weights_grads_sum = np.zeros((self.output_size, self.input_size))
	
	def reset(self):
		self.update(0)
		self.biases = np.random.randn(self.output_size, 1) * TRAINABLE_INIT_SCALE
		self.weights = np.random.randn(self.output_size, self.input_size) * TRAINABLE_INIT_SCALE




# reshape layer
class Reshape(Layer):
	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape
	
	def forward(self, x):
		return np.reshape(x, self.output_shape)
	
	def backward(self, output_delta):
		return np.reshape(output_delta, self.input_shape)




# convolutional layer
class Convolutional2D(Layer):
	def __init__(self, input_shape, kernel_size, kernel_count):
		super().__init__()
		input_depth, input_height, input_width = input_shape
		self.input_shape = input_shape
		self.input_depth = input_depth
		self.kernel_count = kernel_count
		self.kernels_shape = (kernel_count, input_depth, kernel_size, kernel_size)
		self.output_shape = (kernel_count, input_height - kernel_size + 1, input_width - kernel_size + 1)
		
		self.biases = np.random.randn(*self.output_shape) * TRAINABLE_INIT_SCALE
		self.kernels = np.random.randn(*self.kernels_shape) * TRAINABLE_INIT_SCALE
		
		self.biases_grads_sum = np.zeros(self.output_shape)
		self.kernel_grads_sum = np.zeros(self.kernels_shape)
	
	def forward(self, x):
		self.input = x
		self.output = np.copy(self.biases)
		
		for i in range(self.kernel_count):
			for j in range(self.input_depth):
				self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'valid')
		
		return self.output
	
	def backward(self, output_delta):
		biases_delta = output_delta
		kernels_delta = np.zeros(self.kernels_shape)
		input_delta = np.zeros(self.input_shape)
		
		for i in range(self.kernel_count):
			for j in range(self.input_depth):
				kernels_delta[i, j] = signal.correlate2d(self.input[j], output_delta[i], 'valid')
				input_delta[j] += signal.convolve2d(output_delta[i], self.kernels[i, j], 'full')
		
		self.biases_grads_sum += biases_delta
		self.kernel_grads_sum += kernels_delta
		return input_delta
	
	def update(self, eta):
		self.biases -= self.biases_grads_sum * eta
		self.kernels -= self.kernel_grads_sum * eta
		
		self.biases_grads_sum = np.zeros(self.output_shape)
		self.kernel_grads_sum = np.zeros(self.kernels_shape)
	
	def reset(self):
		self.update(0)
		self.biases = np.random.randn(*self.output_shape) * TRAINABLE_INIT_SCALE
		self.kernels = np.random.randn(*self.kernels_shape) * TRAINABLE_INIT_SCALE




# transposed convolutional layer
class ConvolutionalTranspose2D(Convolutional2D):
	def __init__(self, input_shape, kernel_size, kernel_count):
		input_depth, input_height, input_width = input_shape
		self.input_shape = input_shape
		self.input_depth = input_depth
		self.kernel_count = kernel_count
		self.kernels_shape = (kernel_count, input_depth, kernel_size, kernel_size)
		self.output_shape = (kernel_count, input_height + kernel_size - 1, input_width + kernel_size - 1)
		
		self.biases = np.random.randn(*self.output_shape) * TRAINABLE_INIT_SCALE
		self.kernels = np.random.randn(*self.kernels_shape) * TRAINABLE_INIT_SCALE
		
		self.biases_grads_sum = np.zeros(self.output_shape)
		self.kernel_grads_sum = np.zeros(self.kernels_shape)
	
	def forward(self, x):
		self.input = x
		self.output = np.copy(self.biases)
		
		for i in range(self.kernel_count):
			for j in range(self.input_depth):
				self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], 'full')
		
		return self.output
	
	def backward(self, output_delta):
		biases_delta = output_delta
		kernels_delta = np.zeros(self.kernels_shape)
		input_delta = np.zeros(self.input_shape)
		
		for i in range(self.kernel_count):
			for j in range(self.input_depth):
				kernels_delta[i, j] = signal.correlate2d(self.input[j], output_delta[i], 'valid')
				input_delta[j] += signal.convolve2d(output_delta[i], self.kernels[i, j], 'valid')
		
		self.biases_grads_sum += biases_delta
		self.kernel_grads_sum += kernels_delta
		return input_delta




# max pooling layer
class MaxPooling2D(Layer):
	def __init__(self, input_shape, pool_size):
		super().__init__()
		input_depth, input_height, input_width = input_shape
		self.input_depth = input_depth
		self.input_shape = input_shape
		self.input_mask = np.zeros(input_shape)
		
		pool_height, pool_width = pool_size
		self.pool_height = pool_height
		self.pool_width = pool_width
		
		self.output_height = int(input_height / pool_height)
		self.output_width = int(input_width / pool_width)
		self.output_shape = (self.input_depth, self.output_height, self.output_width)
	
	def forward(self, x):
		self.input = x
		self.input_mask = np.zeros(self.input_shape)
		self.output = np.zeros(self.output_shape)
		
		for i in range(self.input_depth):
			for j in range(self.pool_height):
				for k in range(self.pool_width):
					j0 = j*self.pool_height
					j1 = (j+1)*self.pool_height
					k0 = k*self.pool_width
					k1 = (k+1)*self.pool_width
					
					pool = x[i][j0:j1, k0:k1]
					max_y, max_x = np.unravel_index(pool.argmax(), pool.shape)

					self.input_mask[i][j0+max_y][k0+max_x] = 1
					self.output[i][j][k] = pool[max_y][max_x]
		
		return self.output
	
	def backward(self, output_delta):
		input_delta = self.input_mask
		
		for i in range(self.input_depth):
			for j in range(self.output_height):
				for k in range(self.output_width):
					j0 = j*self.pool_height
					j1 = (j+1)*self.pool_height
					k0 = k*self.pool_width
					k1 = (k+1)*self.pool_width
					
					input_delta[i][j0:j1, k0:k1] *= output_delta[i][j][k]
		
		return input_delta




# for padding edges and intra-pixel space of images with zeros 
class ZeroPadding2D(Layer):
	def __init__(self, input_shape, extra_padding, intra_padding):
		self.input_shape = input_shape
		self.extra_padding = extra_padding
		self.intra_padding = intra_padding
		
		self.ep_pattern = ((0,0), (extra_padding,extra_padding), (extra_padding,extra_padding))
		# enable for equal top-left and bottom-right pixel padding 
		#self.ep_pattern = ((0,0), (extra_padding,extra_padding+intra_padding), (extra_padding,extra_padding+intra_padding))
		self.ip_pattern = ((0,0), (intra_padding,0), (intra_padding,0))
		self.ip_kernel = np.pad(np.ones((1,1,1)), self.ip_pattern)
	
	def forward(self, x):
		return np.pad(np.kron(x, self.ip_kernel), self.ep_pattern)
	
	def backward(self, output_delta):
		ep = self.extra_padding
		# enable for equal top-left and bottom-right pixel padding 
		#ep = self.extra_padding + self.intra_padding
		ip = self.intra_padding + 1
		return output_delta[:, ep:-ep or None:ip, ep:-ep or None:ip]



