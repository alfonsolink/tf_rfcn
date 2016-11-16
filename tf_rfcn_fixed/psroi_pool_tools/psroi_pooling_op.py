import tensorflow as tf
import os.path as osp

filename = '/home/alfonso/tensorflow/bazel-bin/tensorflow/core/user_ops/ps_roipool/psroi_pooling.so'
_psroi_pooling_module = tf.load_op_library(filename)
psroi_pool = _psroi_pooling_module.psroi_pool
psroi_pool_grad = _psroi_pooling_module.psroi_pool_grad
