
import os
import sys
import json
import time
import logging
import argparse

import tensorflow as tf
# Get rid of the deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

from models.models import DeepStatisticalSolver 


# Build parser
parser = argparse.ArgumentParser()

# Define mode
parser.add_argument('--infer_data', type=str,
    help='If specified, data on which to evaluate a reloaded model. If specified, you should also specify'\
    +' a result_dir!')

# Define training parameters
parser.add_argument('--rdm_seed', type=int,
    help='Random seed. Random by default.')
parser.add_argument('--gpu', type=int, default=None,
    help='Use GPUs for data generation.')
parser.add_argument('--profile', type=bool, default=False,
    help='Computational graph profiling, for debug purpose.')
parser.add_argument('--max_iter', type=int, default=1000000,
    help='Number of training steps')
parser.add_argument('--minibatch_size', type=int, default=10,
    help='Size of each minibatch')
parser.add_argument('--learning_rate', type=float, default=1e-3,
    help='Learning rate')
parser.add_argument('--discount', type=float, default=0.9,
    help='Discount factor for training')
parser.add_argument('--track_validation', type=float, default=100,
    help='Tracking validation metrics every XX iterations')
parser.add_argument('--data_directory', type=str, default='data/',
    help='Path to the folder containing data')
parser.add_argument('--proxy', action='store_true',
    help='Activates proxy mode')

# Define model parameters
parser.add_argument('--latent_dimension', type=int, default=10,
    help='Dimension of the latent messages, and of the hidden layers of neural net blocks')
parser.add_argument('--hidden_layers', type=int, default=3,
    help='Number of hidden layers in each neural network block')
parser.add_argument('--correction_updates', type=int, default=15,
    help='Number of correction update of the neural network')
parser.add_argument('--alpha', type=float, default=1e-3,
    help='Multiplicative factor for correction updates')
parser.add_argument('--non_linearity', type=str, default='leaky_relu',
    help='Non linearity of the neural network')

# Define directory to store results and models, or to reload from it
parser.add_argument('--result_dir',
    help="Experiment directory. If specified, restores a model.")


if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()

    # Set tensorflow random seed for reproductibility, if defined
    if args.rdm_seed is not None:
        tf.set_random_seed(args.rdm_seed)
        np.random.seed(args.rdm_seed)

    # Select visible GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

    # Setup session
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement=True
    if args.gpu is not None:
        config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Setup results directory
    if args.result_dir is None:
        result_dir = 'results/' + str(int(time.time()))
    else:
        result_dir = args.result_dir

    # Make result directory if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Console
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Log file
    logFile = os.path.join(result_dir, 'model.log')
    handler = logging.FileHandler(logFile, "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt='%Y-%m-%d %H:%M')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # If there is both a model to reload and some data to perform inference on
    if (args.infer_data is not None) and (args.result_dir is not None):

        # Reload the model
        model = DeepStatisticalSolver(args.hidden_dim,
            args.hidden_lay,
            args.correction_update,
            args.non_linearity,
            directory=result_dir,
            model_to_reload=args.result_dir)

        # Evaluate reloaded model on specified data
        loss_test = model.evaluate(mode='test',
            data_directory=args.infer_data)
        logging.info('    Loss on test set : {}'.format(loss_test))

    else:
        
        # Build model
        # If a model_to_reload directory has been specified, then the model will be reloaded
        # and training will start where it last stopped
        model = DeepStatisticalSolver(
            sess,
            latent_dimension=args.latent_dimension,
            hidden_layers=args.hidden_layers,
            correction_updates=args.correction_updates,
            alpha=args.alpha,
            non_lin=args.non_linearity,
            minibatch_size=args.minibatch_size,
            name='gns',
            directory=result_dir,
            default_data_directory=args.data_directory,
            model_to_restore=args.result_dir,
            proxy=args.proxy
        )

        # Train model on the specified directory for data
        model.train(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate, 
            discount=args.discount,
            data_directory=args.data_directory,
            save_step=args.track_validation,
            profile=args.profile)

        # Evaluate the model on validation and test datasets, it also stores predictions
        loss_val = model.evaluate(mode='val',
            data_directory=args.data_directory)
        logging.info('    Loss on validation set : {}'.format(loss_val))

        loss_test = model.evaluate(mode='test',
            data_directory=args.data_directory)
        logging.info('    Loss on test set : {}'.format(loss_test))
