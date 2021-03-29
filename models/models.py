import os
import sys
import json
import time
import copy
import logging

from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from models.layers import FullyConnected, custom_gather, custom_scatter 

class DeepStatisticalSolver:

    def __init__(self,
        sess,
        latent_dimension=10,
        hidden_layers=3,
        correction_updates=5,
        alpha=1e-3,
        non_lin='leaky_relu',
        minibatch_size=10,
        name='deep_statistical_solver',
        directory='./',
        model_to_restore=None,
        default_data_directory='datasets/spring/default',
        proxy=False):

        self.sess = sess
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.correction_updates = correction_updates
        self.alpha = alpha
        self.non_lin = non_lin
        self.minibatch_size = minibatch_size
        self.name = name
        self.directory = directory
        self.current_train_iter = 0
        self.default_data_directory = default_data_directory
        self.proxy = proxy

        # Initialize list of trainable variables
        self.trainable_variables = []

        try:
            # Try importing the dimensions associated to the problem
            sys.path.append(self.default_data_directory)
            from problem import Problem

            self.problem = Problem()

        except ImportError:
            print('You should provide a compatible "problem.py" file in your data folder!')

        # Reload config if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):

            logging.info('    Restoring model from '+model_to_restore)

            # Reload the parameters from a json file
            path_to_config = os.path.join(model_to_restore, 'config.json')
            with open(path_to_config, 'r') as f:
                config = json.load(f)
            self.set_config(config)

        else:
            self.d_in_A = self.problem.d_in_A
            self.d_in_B = self.problem.d_in_B
            self.d_out = self.problem.d_out
            self.d_F = self.problem.d_F
            self.initial_U = self.problem.initial_U

            # Standardization constants
            self.B_mean = self.problem.B_mean
            self.B_std = self.problem.B_std
            self.A_mean = self.problem.A_mean
            self.A_std = self.problem.A_std

            # # Normalization constants
            # self.B_min = self.problem.B_min
            # self.B_max = self.problem.B_max
            # self.A_min = self.problem.A_min
            # self.A_max = self.problem.A_max
        
        # Build weight tensors
        self.build_weights()

        # Build computational graph
        self.build_graph(self.default_data_directory)

        # Restore trained weights if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):

            # Reload the weights from a ckpt file
            saver = tf.compat.v1.train.Saver(self.trainable_variables)
            path_to_weights = os.path.join(model_to_restore, 'model.ckpt')
            saver.restore(self.sess, path_to_weights)

        # Else, randomly initialize weights
        else:
            self.sess.run(tf.compat.v1.variables_initializer(self.trainable_variables))

        # Log config infos
        self.log_config()


    def build_weights(self):
        """
        Builds all the trainable variables
        """

        # Build weights of each correction update block, and store them
        self.psi = {}

        self.phi_from = {}
        self.phi_to = {}
        self.phi_loop = {}

        self.xi = {}

        for update in range(self.correction_updates):

            self.psi[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_correction_block_{}'.format(update),
                input_dim=4*(self.latent_dimension)+self.d_in_B
            )
            self.phi_from[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_phi_from_{}'.format(update),
                input_dim=2*(self.latent_dimension)+self.d_in_A
            )
            self.phi_to[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_phi_to_{}'.format(update),
                input_dim=2*(self.latent_dimension)+self.d_in_A
            )
            self.phi_loop[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_phi_loop_{}'.format(update),
                input_dim=2*(self.latent_dimension)+self.d_in_A
            )
    # decoder
        for update in range(self.correction_updates+1):
            self.xi[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.latent_dimension,
                hidden_layers=self.hidden_layers,#1,
                name=self.name+'_D_{}'.format(update),
                input_dim=self.latent_dimension,
                output_dim=self.d_out
            )
        # self.D = FullyConnected(
        #     non_lin=self.non_lin,
        #     latent_dimension=self.latent_dimension,
        #     hidden_layers=self.hidden_layers,#1,
        #     name=self.name+'_D_{}'.format(update),
        #     input_dim=self.latent_dimension,
        #     output_dim=self.d_out
        # )

        #self.loss_function = self.cost_function EquilibriumViolation(self.default_data_directory)



    def build_graph(self, default_data_directory):
        """
        Builds the computation graph.
        Assumes that all graphs have been merged into one supergraph
        """
        def extract_fn(tfrecord):

            # Extract features using the keys set during creation
            features = {
                'A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                'U': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            }

            # Extract the data record
            sample = tf.parse_single_example(tfrecord, features)

            return [sample['A'], sample['B'], sample['U']]

        self.train_dataset = tf.data.TFRecordDataset([os.path.join(default_data_directory,'train.tfrecords')])
        self.train_dataset = self.train_dataset.map(extract_fn).shuffle(100).batch(self.minibatch_size).repeat()

        self.valid_dataset = tf.data.TFRecordDataset([os.path.join(default_data_directory,'val.tfrecords')])
        self.valid_dataset = self.valid_dataset.map(extract_fn).shuffle(100).batch(self.minibatch_size).repeat()

        # Build iterator
        self.iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(self.train_dataset),
            None)
        self.next_element = self.iterator.get_next()

        # Build operations to initialize training and validation
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(self.valid_dataset)

        # Get the output of the data handler
        self.A_flat, self.B_flat, self.U_flat = self.next_element

        # Reshape the iterator
        self.minibatch_size_ = tf.shape(self.A_flat)[0]
        self.A = tf.reshape(self.A_flat, [self.minibatch_size_, -1, self.d_in_A+2])
        self.B = tf.reshape(self.B_flat, [self.minibatch_size_, -1, self.d_in_B])
        self.U_gt = tf.reshape(self.U_flat, [self.minibatch_size_, -1, self.d_out])

        # Get relevant tensor dimensions
        self.minibatch_size_tf = tf.shape(self.A)[0]
        self.num_nodes = tf.shape(self.B)[1]
        self.num_edges = tf.shape(self.A)[1]
        self.A_dim = tf.shape(self.A)[2]

        # Getting standardization constants
        self.A_mean_tf = tf.ones([self.minibatch_size_tf, self.num_edges, 1]) * \
            tf.reshape(tf.constant(self.A_mean, dtype=tf.float32), [1, 1, 2+self.d_in_A])
        self.A_std_tf = tf.ones([self.minibatch_size_tf, self.num_edges, 1]) * \
            tf.reshape(tf.constant(self.A_std, dtype=tf.float32), [1, 1, 2+self.d_in_A])
        self.B_mean_tf = tf.ones([self.minibatch_size_tf, self.num_nodes, 1]) * \
            tf.reshape(tf.constant(self.B_mean, dtype=tf.float32), [1, 1, self.d_in_B])
        self.B_std_tf = tf.ones([self.minibatch_size_tf, self.num_nodes, 1]) * \
            tf.reshape(tf.constant(self.B_std, dtype=tf.float32), [1, 1, self.d_in_B])

        # Standardizing inputs A and B. Lower case means normalized
        self.a = (self.A - self.A_mean_tf) / self.A_std_tf
        self.b = (self.B - self.B_mean_tf) / self.B_std_tf

        # # Getting normalization constants
        # self.A_min_tf = tf.ones([self.minibatch_size_tf, self.num_edges, 1]) * \
        #     tf.reshape(tf.constant(self.A_min, dtype=tf.float32), [1, 1, 2+self.d_in_A])
        # self.A_max_tf = tf.ones([self.minibatch_size_tf, self.num_edges, 1]) * \
        #     tf.reshape(tf.constant(self.A_max, dtype=tf.float32), [1, 1, 2+self.d_in_A])
        # self.B_min_tf = tf.ones([self.minibatch_size_tf, self.num_nodes, 1]) * \
        #     tf.reshape(tf.constant(self.B_min, dtype=tf.float32), [1, 1, self.d_in_B])
        # self.B_max_tf = tf.ones([self.minibatch_size_tf, self.num_nodes, 1]) * \
        #     tf.reshape(tf.constant(self.B_max, dtype=tf.float32), [1, 1, self.d_in_B])
        # # Normalizing input A and B
        # self.a = (self.A - self.A_min_tf) / (self.A_max_tf - self.A_min_tf)
        # self.b = (self.B - self.B_min_tf) / (self.B_max_tf - self.B_min_tf)

        # Extract indices from matrix A (indices are indeed not normalized)
        self.indices_from = tf.cast(self.A[:,:,0], tf.int32)
        self.indices_to = tf.cast(self.A[:,:,1], tf.int32)

        # Build mask to detect loops
        self.mask_loop = tf.cast(tf.math.equal(self.indices_from, self.indices_to), tf.float32)
        self.mask_loop = tf.expand_dims(self.mask_loop, -1)

        # Extract normalized edge characteristics from matrix A
        self.a_ij = self.a[:,:,2:]

        # Initialize the discount factor that will later be updated
        self.discount = tf.Variable(0., trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.discount]))

        # Initialize messages, predictions and losses dict
        self.H = {}
        self.U = {}
        self.loss = {}
        self.loss_proxy = {}
        self.log_loss = {}
        self.cost_per_sample = {}
        self.total_loss = None
        self.total_loss_proxy = None

        # Get the natural offset that will be added to every output U at every node
        self.initial_U_tf = tf.ones([self.minibatch_size_tf, self.num_nodes, 1]) * \
            tf.reshape(tf.constant(self.initial_U, dtype=tf.float32), [1, 1, self.d_out])

        # # Initialize latent message and prediction to 0
        self.H['0'] = tf.zeros([self.minibatch_size_tf, self.num_nodes, self.latent_dimension])

        # Initialize latent message and prediction to a random tensor
        # self.H['0'] = tf.random.uniform([self.minibatch_size_tf, self.num_nodes, self.latent_dimension])
        # Decode the first message. Although this step useless, it is still there for compatibility issues
        self.U['0'] = self.xi['0'](self.H['0']) + self.initial_U_tf

        # Iterate over every correction update (k_bar)
        for update in range(self.correction_updates):
            # Gather messages from both extremities of each edges
            self.H_from = custom_gather(self.H[str(update)], self.indices_from)
            self.H_to = custom_gather(self.H[str(update)], self.indices_to)

            # Concatenate all the inputs of the phi neural network
            self.Phi_input = tf.concat([self.H_from, self.H_to, self.a_ij], axis=2)

            # Compute the phi using the dedicated neural network blocks
            self.Phi_from = self.phi_from[str(update)](self.Phi_input) * (1.-self.mask_loop)
            self.Phi_to = self.phi_to[str(update)](self.Phi_input) * (1.-self.mask_loop)
            self.Phi_loop = self.phi_loop[str(update)](self.Phi_input) * self.mask_loop

            # Get the sum of each transformed messages at each node
            self.Phi_from_sum = custom_scatter(
                self.indices_from, 
                self.Phi_from, 
                [self.minibatch_size_tf, self.num_nodes, self.latent_dimension])
            self.Phi_to_sum = custom_scatter(
                self.indices_to, 
                self.Phi_to, 
                [self.minibatch_size_tf, self.num_nodes, self.latent_dimension])
            self.Phi_loop_sum = custom_scatter(
                self.indices_to, 
                self.Phi_loop, 
                [self.minibatch_size_tf, self.num_nodes, self.latent_dimension])

            # Concatenate all the inputs of the correction neural network
            self.correction_input = tf.concat([
                self.H[str(update)],
                self.Phi_from_sum,
                self.Phi_to_sum,
                self.Phi_loop_sum,
                self.b], axis=2)


            # Compute the correction using the dedicated neural network block
            self.correction = self.psi[str(update)](self.correction_input)

            # Apply correction, and extract the predictions from the latent message
            self.H[str(update+1)] = self.H[str(update)] + self.correction * self.alpha

            # Decode H
            self.U[str(update+1)] = self.xi[str(update+1)](self.H[str(update+1)]) + self.initial_U_tf

            # Compute the loss for each sample
            # if self.proxy:
            self.loss_proxy[str(update + 1)] = \
                    tf.reduce_mean((self.U[str(update+1)] - self.U_gt)**2)
            self.cost_per_sample[str(update+1)] = \
                self.problem.cost_function(self.U[str(update+1)], self.A, self.B)

            # Take the mean, and register its value for tensorboard
            self.loss[str(update+1)] = tf.reduce_mean(self.cost_per_sample[str(update+1)])
            tf.compat.v1.summary.scalar("loss_{}".format(update+1), self.loss[str(update+1)])
            # if self.proxy:
            tf.compat.v1.summary.scalar("loss_proxy_{}".format(update+1), self.loss_proxy[str(update+1)])

            # Compute the discounted loss
            if self.total_loss is None:
                self.total_loss = self.loss[str(update+1)] * self.discount**(self.correction_updates-1-update)
                # if self.proxy:
                self.total_loss_proxy = self.loss_proxy[str(update+1)] * \
                                            self.discount**(self.correction_updates-1-update)
            else:
                self.total_loss += self.loss[str(update+1)] * self.discount**(self.correction_updates-1-update)
                # if self.proxy:
                self.total_loss_proxy += self.loss_proxy[str(update+1)] * \
                                             self.discount**(self.correction_updates-1-update)

        # Get the final prediction and the final loss
        self.U_final = self.U[str(self.correction_updates)]
        if self.proxy:
            self.loss_final = self.loss_proxy[str(self.correction_updates)]
        else:
            self.loss_final = self.loss[str(self.correction_updates)]
        

        # Initialize the optimizer
        self.learning_rate = tf.Variable(0., trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.learning_rate]))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)

        # Calculate gradient of trainable variables w.r.t. the specified loss function
        if self.proxy:
            self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.total_loss_proxy))
        else:
            self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.total_loss))
        
        # Gradient clipping to avoid exploding gradients
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1)
        # Define optimization operator
        self.opt_op = self.optimizer.apply_gradients(zip(self.gradients, self.variables))

        # Initialize training variables
        self.sess.run(tf.compat.v1.variables_initializer(self.optimizer.variables()))

        # Build summary to visualize the final loss in Tensorboard
        tf.compat.v1.summary.scalar("loss_final", self.loss_final)
        self.merged_summary_op = tf.compat.v1.summary.merge_all()

        # Gather trainable variables
        for update in range(self.correction_updates):
            self.trainable_variables.extend(self.phi_from[str(update)].trainable_variables)
            self.trainable_variables.extend(self.phi_to[str(update)].trainable_variables)
            self.trainable_variables.extend(self.phi_loop[str(update)].trainable_variables)
            self.trainable_variables.extend(self.psi[str(update)].trainable_variables)
        for update in range(self.correction_updates+1):
            self.trainable_variables.extend(self.xi[str(update)].trainable_variables)


    def log_config(self):
        """
        Logs the config of the whole model
        """

        logging.info('    Configuration :')
        logging.info('        Storing model in  : '+self.directory)
        logging.info('        Latent dimensions : {}'.format(self.latent_dimension))
        logging.info('        Number of hidden layers per block : {}'.format(self.hidden_layers))
        logging.info('        Number of correction updates : {}'.format(self.correction_updates))
        logging.info('        Alpha : {}'.format(self.alpha))
        logging.info('        Non linearity : {}'.format(self.non_lin))
        logging.info('        d_in_A : {}'.format(self.d_in_A))
        logging.info('        d_in_B : {}'.format(self.d_in_B))
        logging.info('        d_out : {}'.format(self.d_out))
        #logging.info('        d_F : {}'.format(self.d_F))
        logging.info('        Minibatch size : {}'.format(self.minibatch_size))
        logging.info('        Current training iteration : {}'.format(self.current_train_iter))
        logging.info('        Model name : ' + self.name)
        logging.info('        Initial U : {}'.format(self.initial_U))
        logging.info('        A mean : {}'.format(self.A_mean))
        logging.info('        A std : {}'.format(self.A_std))
        logging.info('        B mean : {}'.format(self.B_mean))
        logging.info('        B std : {}'.format(self.B_std))
        # logging.info('        A min : {}'.format(self.A_min))
        # logging.info('        A max : {}'.format(self.A_max))
        # logging.info('        B min : {}'.format(self.B_min))
        # logging.info('        B max : {}'.format(self.B_max))
        logging.info('        Proxy : {}'.format(self.proxy))

    def set_config(self, config):
        """
        Sets the config according to an inputed dict
        """

        self.latent_dimension = config['latent_dimension']
        self.hidden_layers = config['hidden_layers']
        self.correction_updates = config['correction_updates']
        self.alpha = config['alpha']
        self.non_lin = config['non_lin']
        self.d_in_A = config['d_in_A']
        self.d_in_B = config['d_in_B']
        self.d_out = config['d_out']
        #self.d_F = config['d_F']
        self.minibatch_size = config['minibatch_size']
        self.name = config['name']
        self.directory = config['directory']
        self.current_train_iter = config['current_train_iter']
        self.initial_U = np.array(config['initial_U'])
        self.A_mean = np.array(config['A_mean'])
        self.A_std = np.array(config['A_std'])
        self.B_mean = np.array(config['B_mean'])
        self.B_std = np.array(config['B_std'])
        # self.A_min = np.array(config['A_min'])
        # self.A_max = np.array(config['A_max'])
        # self.B_min = np.array(config['B_min'])
        # self.B_max = np.array(config['B_max'])
        if 'proxy' in config:
            self.proxy = config['proxy']
        else:
            self.proxy = False

    def get_config(self):
        """
        Gets the config dict
        """

        config = {
            'latent_dimension': self.latent_dimension,
            'hidden_layers': self.hidden_layers,
            'correction_updates': self.correction_updates,
            'alpha': self.alpha,
            'non_lin': self.non_lin,
            'd_in_A': self.d_in_A,
            'd_in_B': self.d_in_B,
            'd_out': self.d_out,
            #'d_F': self.d_F,
            'minibatch_size': self.minibatch_size,
            'name': self.name,
            'directory': self.directory,
            'current_train_iter': self.current_train_iter,
            'initial_U': list(self.initial_U),
            'A_mean': list(self.A_mean),
            'A_std': list(self.A_std),
            'B_mean': list(self.B_mean),
            'B_std': list(self.B_std),
            # 'A_min': list(self.A_min),
            # 'A_max': list(self.A_max),
            # 'B_min': list(self.B_min),
            # 'B_max': list(self.B_max),
            'proxy': self.proxy
        } 
        return config

    def save(self):
        """
        Saves the configuration of the model and the trained weights
        """

        # Save config
        config = self.get_config()
        path_to_config = os.path.join(self.directory, 'config.json')
        with open(path_to_config, 'w') as f:
            json.dump(config, f)

        # Save weights
        saver = tf.compat.v1.train.Saver(self.trainable_variables)
        path_to_weights = os.path.join(self.directory, 'model.ckpt')
        saver.save(self.sess, path_to_weights)


    def train(self, 
        max_iter=10,
        learning_rate=3e-4, 
        discount=0.9,
        data_directory='datasets/spring/default',
        save_step=None,
        profile=False):
        """
        Performs a training process while keeping track of the validation score
        """

        # Log infos about training process
        logging.info('    Starting a training process :')
        logging.info('        Max iteration : {}'.format(max_iter))
        logging.info('        Learning rate : {}'.format(learning_rate))
        logging.info('        Discount : {}'.format(discount))
        logging.info('        Training data : {}'.format(data_directory))
        logging.info('        Saving model every {} iterations'.format(save_step))
        if profile:
            logging.info('        Profiling...')

        # Load dataset
        self.sess.run(self.training_init_op)

        # Build writer dedicated to training for Tensorboard
        self.training_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(self.directory, 'train'))

        # Build writer dedicated to validation for Tensorboard
        self.validation_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(self.directory, 'val'))

        # Set discount factor and learning rate
        self.sess.run(self.discount.assign(discount))
        self.sess.run(self.learning_rate.assign(learning_rate))

        # Copy the latest training iteration of the model
        starting_point = copy.copy(self.current_train_iter)

        # If profiling, then initialize useful variables
        if profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            profile_path = os.path.join(self.directory, 'profile')

        # Training loop
        for i in tqdm(range(starting_point, starting_point+max_iter)):

            # Store current training step, so that it's always up to date
            self.current_train_iter = i

            # Perform SGD step
            if profile and i>starting_point:
                self.sess.run(self.opt_op, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(profile_path+'_{}.json'.format(i), 'w') as f:
                    f.write(chrome_trace)
            else:
                self.sess.run(self.opt_op)

            # Store final loss in a summary
            self.summary = self.sess.run(self.merged_summary_op)
            self.training_writer.add_summary(self.summary, self.current_train_iter)

            # Periodically log metrics and save model
            if ((save_step is not None) & (i % save_step == 0)) or (i == starting_point+max_iter-1):

                # Get minibatch train loss
                loss_final_train = self.sess.run(self.loss_final)

                # Change source data to validation
                self.sess.run(self.validation_init_op)

                # Get minibatch val loss
                loss_final_val = self.sess.run(self.loss_final)

                # Store final loss in validation
                self.summary = self.sess.run(self.merged_summary_op)
                self.validation_writer.add_summary(self.summary, self.current_train_iter)

                # Change source data to validation
                self.sess.run(self.training_init_op)

                # Log metrics
                logging.info('    Learning iteration {}'.format(i))
                logging.info('    Training loss (minibatch) : {}'.format(loss_final_train))
                logging.info('    Validation loss (minibatch): {}'.format(loss_final_val))

                # Save model
                self.save()

        # Save model at the end of training
        self.save()

    def evaluate(self,
        mode='val',
        data_directory='data'):
        """
        Evaluate loss on the desired dataset and stores predictions
        """

        # Import numpy dataset
        data_plot = 'datasets/spring/large'
        data_file = '_test.npy'
        A = np.load(os.path.join(data_directory, 'A_'+mode+'.npy'))
        B = np.load(os.path.join(data_directory, 'B_'+mode+'.npy'))

        # Compute final loss
        loss = self.sess.run(self.loss_final, feed_dict={self.A:A, self.B:B})

        # Compute and save the final prediction
        X_final = self.sess.run(self.X_final, feed_dict={self.A:A, self.B:B})
        np.save(os.path.join(self.directory, 'X_final_pred_'+mode+'.npy'), X_final)

        return loss









