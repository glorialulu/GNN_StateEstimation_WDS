# This file defines the interaction and counteraction forces, as well as their first order derivatives
import tensorflow as tf
import numpy as np
import sys

def custom_gather(params, indices_edges):
    """
    This computational graph module performs the gather_nd operation while taking into account
    the batch dimension.

    Inputs
        - params : tf tensor of shape [n_samples, n_nodes, d_out], and type tf.float64
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
    Output
        - tf tensor of shape [n_samples, n_edges, d_out] and type tf.float64
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
    params_flat = tf.reshape(params, [-1, d_out])                       # tf.float64, [n_samples * n_nodes, d_out]

    # Perform the gather operation
    gathered_flat = tf.gather_nd(params_flat, indices_flat)             # tf.float64, [n_samples * n_edges, d_out]

    # Un-flatten the result of the gather operation
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])   # [n_samples , n_edges, d_out]

    return gathered

def custom_scatter(indices_edges, params, shape):
    """
    This computational graph module performs the scatter_nd operation while taking into account
    the batch dimension. Note that here we can also have d instead of d_F

    Inputs
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - params : tf tensor of shape [n_samples, n_edges, d_F], and type tf.float64
        - shape : tf.tensor of shape [3]
    Output
        - tf tensor of shape [n_samples, n_nodes, n_nodes, d_F] and type tf.float64
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



class Problem:

    def __init__(self):

        self.name = 'WDS Hydraulic Simulation'
        
        # Input dimensions
        self.d_in_A = 1
        self.d_in_B = 4

        # Output dimensions
        self.d_out = 1

        # How many equations should how for each node
        self.d_F = 1

        self.initial_U = np.array([399.])

        # Standardization constants
        self.B_mean = np.array([0.0,  0.005844823156379753, 0.0, 15.621948897809508])
        self.B_std = np.array([1.0,  0.004587796037337935, 1.0, 77.32483667915464])
        self.A_mean = np.array([0.0, 0.0, 0.08255444886402044])
        self.A_std = np.array([1.0, 1.0, 0.17580375469346807])

    def cost_function(self, U, A, B):
        # Check input
        U = tf.debugging.check_numerics( U,'U is NaN')
        A = tf.debugging.check_numerics( A,'A is NaN')
        B = tf.debugging.check_numerics( B,'B is NaN')

        # Gather instances dimensions (samples, nodes and edges)
        n_samples = tf.shape(U)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(U)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]
        # Extact edge characteristics from A matrix
        A_ij = A[:,:,2:3]                                          # tf.float64, [n_samples, n_edge, d_in_A]

        # demand node indicator (1:junctions; 0:reservoirs/tanks)
        Nd = B[:,:,0:1]

        # demand at junctions
        demand = B[:,:,1:2]  # [L/s]

        # head boundary indicator(1: head unknown, 0: head known)
        Nh = B[:,:,2:3]

        # known head
        source_head =B[:,:,3:4]       #[m]

        # Gather head (pressure+elevation) on both sides of each edge
        # H_i = custom_gather(Nh*U[:,:,0:1] + (1-Nh)*source_head, indices_from)      # tf.float64, [n_samples , n_edges, d_out]
        # H_j = custom_gather(Nh*U[:,:,0:1] + (1-Nh)*source_head, indices_to)        # tf.float64, [n_samples , n_edges, d_out]
        H_i = custom_gather(Nd*U[:,:,0:1] + (1-Nd)*source_head, indices_from)      # tf.float64, [n_samples , n_edges, d_out]
        H_j = custom_gather(Nd*U[:,:,0:1] + (1-Nd)*source_head, indices_to)        # tf.float64, [n_samples , n_edges, d_out]

        # Compute head difference
        H_ij = H_i - H_j   # tf.float64, [n_samples, n_edges, 1]


        n = tf.constant(1/1.852, dtype=tf.float32)
        # Compute flow
        Q_ij = tf.sign(H_ij) * tf.pow( tf.maximum(tf.abs(H_ij), 1e-9)*A_ij, n)  #[L/s], [n_samples, n_edges, 1]

        # nodal demand imbalance
        delta_Q =  Nd * ( -demand - custom_scatter(indices_from, Q_ij, [n_samples, n_nodes, 1])
                                + custom_scatter(indices_to, Q_ij, [n_samples, n_nodes, 1])) **2 # tf.float64, [n_samples, n_nodes, 1]

        # difference in reservoir head
        delta_H = (1-Nh) * (U[:,:,0:1] - source_head)**2 # tf.float64, [n_samples, n_nodes, 1]

        cost_per_sample =  tf.reduce_mean(delta_Q, axis=[1,2]) + tf.reduce_mean(delta_H, axis=[1,2]) # tf.float64, [n_samples]

        return cost_per_sample


# train model
# python main.py --data_dir=datasets/asnet2_enforce5_b4 --learning_rate=1e-3 --minibatch_size=500 --alpha=1e-2 --hidden_layers=2 --latent_dimension=20 --correction_updates=20 --track_validation=1000














