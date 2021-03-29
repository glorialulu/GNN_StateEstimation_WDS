#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:21:51 2020

@author: luxing
"""
import tensorflow as tf
import numpy as np
import tqdm
import os 

def np_to_tfrecords(A, B, U, file_path_prefix, verbose=True):
    """
    author : "Sangwoong Yoon"
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    
    assert isinstance(B, np.ndarray)
    assert len(B.shape) == 2
    
    assert isinstance(U, np.ndarray)
    assert len(U.shape) == 2
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_a = _dtype_feature(A)
    dtype_feature_b = _dtype_feature(B)
    dtype_feature_u = _dtype_feature(U)      
        
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(U.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in tqdm.tqdm(range(A.shape[0])):
        a = A[idx]
        b = B[idx]
        u = U[idx]
        
        d_feature = {}
        d_feature['A'] = dtype_feature_a(a)
        d_feature['B'] = dtype_feature_b(b)
        d_feature['U'] = dtype_feature_u(u)
        
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

data_dir = 'datasets/asnet5_5.11.32.37.44/'
# convert to tensorflow records  
for mode in ['train', 'val', 'test']:

    A = np.load(os.path.join(data_dir, 'A_'+mode+'.npy'), allow_pickle=True)
    B = np.load(os.path.join(data_dir, 'B_'+mode+'.npy'), allow_pickle=True)
    U = np.load(os.path.join(data_dir, 'U_'+mode+'.npy'), allow_pickle=True)

    n_samples = np.array(np.shape(A)[0])

    A = np.reshape(A, [n_samples, -1])
    B = np.reshape(B, [n_samples, -1])
    U = np.reshape(U, [n_samples, -1])

    np_to_tfrecords(A, B, U, os.path.join(data_dir, mode), 
        verbose=True)
