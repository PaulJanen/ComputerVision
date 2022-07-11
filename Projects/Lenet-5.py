
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

num_classes = 10
img_rows, img_cols, img_ch = 28, 28, 1
input_shape = (img_rows, img_cols, img_ch)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)


class SimpleConvolutionLayer(tf.keras.layers.Layer):

    def __init__(self, num_kernels=32, kernel_size=(3, 3), strides=(1, 1), use_bias=True):
        """
        Initialize the layer.
        :param num_kernels:     Number of kernels for the convolution
        :param kernel_size:     Kernel size (H x W)
        :param strides:         Vertical and horizontal stride as list
        :param use_bias:        Flag to add a bias after covolution / before activation
        """
        # First, we have to call the `Layer` super __init__(), as it initializes hidden mechanisms:
        super().__init__()  
        # Then we assign the parameters:
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias

    def build(self, input_shape):
        """
        Build the layer, initializing its parameters according to the input shape.
        This function will be internally called the first time the layer is used, though
        it can also be manually called.
        :param input_shape:     Input shape the layer will receive (e.g. B x H x W x C). B stands for batch
        """
        # We are provided with the input shape here, so we know the number of input channels:
        num_input_channels = input_shape[-1]  # assuming shape format BHWC

        # Now we know how the shape of the tensor representing the kernels should be:
        kernels_shape = (*self.kernel_size, num_input_channels, self.num_kernels)

        # For this example, we initialize the filters with values picked from a Glorot distribution:
        glorot_uni_initializer = tf.initializers.GlorotUniform()
        self.kernels = self.add_weight(name='kernels',
                                       shape=kernels_shape,
                                       initializer=glorot_uni_initializer,
                                       trainable=True)  # and we make the variable trainable.

        if self.use_bias:  # If bias should be added, we initialize its variable too:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.num_kernels,),
                                        initializer='random_normal',  # e.g., using normal distribution.
                                        trainable=True)

    def call(self, inputs):
        """
        Call the layer and perform its operations on the input tensors
        :param inputs:  Input tensor
        :return:        Output tensor
        """
        # We perform the convolution:
        z = tf.nn.conv2d(inputs, self.kernels, strides=[1, *self.strides, 1], padding='VALID')

        if self.use_bias:  # we add the bias if requested:
            z = z + self.bias
        # Finally, we apply the activation function (e.g. ReLU):
        return tf.nn.relu(z)

    def get_config(self):
        """
        Helper function to define the layer and its parameters.
        :return:        Dictionary containing the layer's configuration
        """
        return {'num_kernels': self.num_kernels,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'use_bias': self.use_bias}

class LeNet5(Model):
    
    def __init__(self, num_classes):
        """
        Initialize the model.
        :param num_classes:     Number of classes to predict from
        """
        super(LeNet5, self).__init__()
        # We instantiate the various layers composing LeNet-5:
        # self.conv1 = SimpleConvolutionLayer(6, kernel_size=(5, 5))
        # self.conv2 = SimpleConvolutionLayer(16, kernel_size=(5, 5))
        # ... or using the existing and (recommended) Conv2D class:
        self.conv1 = Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu')
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')
        self.max_pool = MaxPooling2D(pool_size=(2, 2)) # only one, because it doesnt have weights and derivative
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        """
        Call the layers and perform their operations on the input tensors
        :param inputs:  Input tensor
        :return:        Output tensor
        """
        x = self.max_pool(self.conv1(inputs))        # 1st block
        x = self.max_pool(self.conv2(x))             # 2nd block
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x))) # dense layers
        return x

model = LeNet5(num_classes)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# We can call `model.summary()` only if the model was built before. 
# It is normally done automatically at the first use of the network,
# inferring the input shapes from the samples the network is given.
# For instance, the command below would build the network (then use it for prediction):
#_ = model.predict(x_test[:10])

# But we can build the model manually otherwise, providing the batched
# input shape ourselves:
batched_input_shape = tf.TensorShape((None, *input_shape))
model.build(input_shape=batched_input_shape)

# Method to visualize the architecture of the network:
model.summary()

callbacks = [
    # Callback to interrupt the training if the validation loss (`val_loss`) stops improving for over 3 epochs:
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    # Callback to log the graph, losses and metrics into TensorBoard (saving log files in `./logs` directory):
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)]

history = model.fit(x_train, y_train,
                    batch_size=32, epochs=80, validation_data=(x_test, y_test), 
                    verbose=2,  # change to `verbose=1` to get a progress bar
                                # (we opt for `verbose=2` here to reduce the log size)
                    callbacks=callbacks)







