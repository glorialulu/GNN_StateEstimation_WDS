import sys
import os

import tensorflow as tf

def custom_gather(params, indices_edges):
    """
    This computational graph module performs the gather_nd operation while taking into account
    the batch dimension.

    Inputs
        - params : tf tensor of shape [n_samples, n_nodes, d_out], and type tf.float32
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
    Output
        - tf tensor of shape [n_samples, n_edges, d_out] and type tf.float32
    """

    # Get all relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = tf.shape(params)[1]                                       # tf.int32, [1]
    n_edges = tf.shape(indices_edges)[1]                                # tf.int32, [1]
    d_out = tf.shape(params)[2]                                         # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Flatten the indices
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the node parameters
    params_flat = tf.reshape(params, [-1, d_out])                       # tf.float32, [n_samples * n_nodes, d_out]

    # Perform the gather operation
    gathered_flat = tf.gather_nd(params_flat, indices_flat)             # tf.float32, [n_samples * n_edges, d_out]

    # Un-flatten the result of the gather operation
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])   # [n_samples , n_edges, d_out]

    return gathered

def custom_scatter(indices_edges, params, shape):
    """
    This computational graph module performs the scatter_nd operation while taking into account
    the batch dimension. Note that here we can also have d instead of d_F

    Inputs
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - params : tf tensor of shape [n_samples, n_edges, d_F], and type tf.float32
        - shape : tf.tensor of shape [3]
    Output
        - tf tensor of shape [n_samples, n_nodes, n_nodes, d_F] and type tf.float32
    """

    # Get all the relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = shape[1]                                                  # tf.int32, [1]
    n_edges = tf.shape(params)[1]                                       # tf.int32, [1]
    d_F = tf.shape(params)[2]                                           # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Stack batch and edge dimensions
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the edge parameters
    params_flat = tf.reshape(params, [n_samples*n_edges, d_F])          # tf.float32, [n_samples * n_edges, d_F]

    # Perform the scatter operation
    scattered_flat = tf.scatter_nd(indices_flat, params_flat, shape=[n_samples*n_nodes, d_F])   
                                                                        # tf.float32, [n_samples * n_nodes, d_F]

    # Un-flatten the result of the scatter operation
    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, d_F])   # tf.float32, [n_samples, n_nodes, d_F]

    return scattered

class FullyConnected:
    """
    Simple fully connected block. Serves as an elementary learning block in our neural network architecture.

    Params
        - latent_dimension : integer, number of hidden neurons in every intermediate layer
        - hidden : integer, number of layers. If set to 1, there is no hidden layer
        - non_lin : string, chosen non linearity
        - input_dim : integer, dimension of the input; if not specified, set to latent_dimension
        - output_dim : integer, dimension of the output; if not specified, set to latent_dimension
        - name : string, name of the neural network block
    """
    
    def __init__(self, 
        latent_dimension=10,
        hidden_layers=3,
        non_lin='leaky_relu', 
        input_dim=None,
        output_dim=None, 
        name='encoder'):
        
        # Get parameters
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize list of trainable variables of the layer
        self.trainable_variables = []
        
        # Convert str into an actual tensorflow operator
        if non_lin == 'tanh':
            self.non_lin = tf.tanh
        elif non_lin == 'leaky_relu':
            self.non_lin = tf.nn.leaky_relu

        # Build weights
        self.build()
        
    def build(self):
        """
        Builds and collects the weights of the neural network block.
        """
        
        # Initialize weights dict
        self.W = {}
        self.b = {} 

        # Iterate over all layers
        for layer in range(self.hidden_layers):

            # Make sure the dimensions are used for the weights
            left_dim = self.latent_dimension
            right_dim = self.latent_dimension
            if (layer == 0) and (self.input_dim is not None):
                left_dim = self.input_dim
            if (layer == self.hidden_layers-1) and (self.output_dim is not None):
                right_dim = self.output_dim

            # Initialize weight matrix
            self.W[str(layer)] = tf.compat.v1.get_variable(name='W_'+self.name+'_{}'.format(layer),
                shape=[left_dim, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.W[str(layer)])

            # Initialize bias vector
            self.b[str(layer)] = tf.compat.v1.get_variable(name='b_'+self.name+'_{}'.format(layer),
                shape=[1, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.b[str(layer)])

                            
    def __call__(self, h):
        """
        Builds the computational graph.
        """
        n_samples = tf.shape(h)[0]
        n_elem = tf.shape(h)[1]
        d = tf.shape(h)[2]

        h = tf.reshape(h, [-1, d])

        for layer in range(self.hidden_layers):
            # Iterate over all layers

            if layer==self.hidden_layers-1:
                # If last layer, then do not apply any non linearity
                h = tf.matmul(h, self.W[str(layer)]) + self.b[str(layer)]

            else:
                h = self.non_lin(tf.matmul(h, self.W[str(layer)])+ self.b[str(layer)])

        return tf.reshape(h, [n_samples, n_elem, -1])

