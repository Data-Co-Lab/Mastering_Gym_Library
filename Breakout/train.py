

class ImageTransform:

	def __init__(self):
		# This is our future input
		self.input_state= 
		output = tf.image.rgb_to_grayscale(state)
		output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
		output = tf.image.resize(output,[32, 32],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		output = tf.squeeze(output)

	def prepare(state) :
		
		array = K.eval(output)
		return np.reshape(array, (1, 32,32,1))
		