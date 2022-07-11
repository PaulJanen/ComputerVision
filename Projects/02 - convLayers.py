import tensorflow as tf



# Defining our convolutional layer as a compiled function:
@tf.function
def conv_layer(x, kernels, bias, s):
    z = tf.nn.conv2d(x, kernels, strides=[1,s,s,1], padding='VALID')
    # Finally, applying the bias and activation function (for instance,ReLU):
    return tf.nn.relu(z + bias)


class SimpleConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, num_kernels=32, kernel_size=(3, 3), stride=1):
        """ Initialize the layer.
        :param num_kernels: Number of kernels for the convolution
        :param kernel_size: Kernel size (H x W)
        :param stride: Vertical/horizontal stride
        """
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, input_shape):
        """
        Build the layer, initializing its parameters/variables.
        This will be internally called the 1st time the layer is used.
        :param input_shape: Input shape for the layer (for instance, BxHxWxC)
        """
        num_input_ch = input_shape[-1] # assuming shape format BHWC
        # Now we know the shape of the kernel tensor we need:
        kernels_shape = (*self.kernel_size, num_input_ch, self.num_kernels)
        # We initialize the filter values fior instance, from a Glorot distribution:
        glorot_init = tf.initializers.GlorotUniform()
        self.kernels = self.add_weight( # method to add Variables to layer
        name='kernels', shape=kernels_shape, initializer=glorot_init,
        trainable=True) # and we make it trainable.
        # Same for the bias variable (for instance, from a normal distribution):
        self.bias = self.add_weight(
        name='bias', shape=(self.num_kernels,),
        initializer='random_normal', trainable=True)

    def call(self, inputs):
        """ Call the layer, apply its operations to the input tensor."""
        return conv_layer(inputs, self.kernels, self.bias, self.stride)


### Or we could simplify whole code into one line:

conv = tf.keras.layers.Conv2D(filters=N, kernel_size=(k, k), strides=s,padding='valid', activation='relu')