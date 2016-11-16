import tensorflow as tf
import numpy as np
import collections
from rpn_tools.anchor_target_layer import *
from rpn_tools.proposal_layer import *
from rpn_tools.proposal_target_layer import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import csv
np.set_printoptions(threshold=np.inf)
import cv2
import glob
from psroi_pool_tools import psroi_pooling_op
from psroi_pool_tools import psroi_pooling_op_grad
from rpn_tools.mnc_data_layer import MNCDataLayer
from cnn_tools.tools import *
import cPickle


# hyperparameter settings
resnet_length = 101 # resnet lengths can be {50, 101, 152}
_tr_bn = True # train BN parameters
_add_l2 = True # include weight decay of 1e-4: set to False
_rpn_stat = True # train RPN weights
_rcnn_stat = True # train RCNN weights
_fc_stat = True # train "fc" -- which are hconv5 residual layers and up
_layer0_stat = False # train hconv2
_layer1_stat = True # train hconv3
_layer2_stat = True # train hconv4
lr_w = 0.001 # learning rate for weights
lr_b = 0.002 # learning rate for biases
s1 = 3 # smoothL1 loss sigma hyperparameter for RPN anchor bbox_pred
s2 = 1 # smoothL1 loss sigma hyperparameter for classification bbox_pred

# other hyperparameter settings: modify with care...
A = 9
rpn_size = 256
rpn_batch_size = 128

# input cache roidb pkl file
cache_file = './data_pkl/mnc_roidb.pkl'

with open(cache_file, 'rb') as fid:
    roidb = cPickle.load(fid)
mnc = MNCDataLayer()
mnc.setup(roidb)


def anchor(x, g, i):
  x = np.array(x)
  g = np.array(g)
  labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, n = \
      forward_anchor_op(x, g, i)
  return labels.astype(np.int64), bbox_targets, bbox_inside_weights, \
      bbox_outside_weights, np.array(n).astype(np.float32)

def proposal(cls, bbox, i):
  cls = np.array(cls)
  bbox = np.array(bbox)
  blob = forward_proposal_op(cls, bbox, i)
  return blob

def proposal_target(rpn_rois, gt):
  rpn_rois = np.array(rpn_rois)
  gt = np.array(gt)
  rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      forward_proposal_target_op(rpn_rois, gt)
  return rois.astype(np.int32), labels.astype(np.int64), bbox_targets, \
      bbox_inside_weights, bbox_outside_weights

def resnet(inpt, n, loc):
  if n == 50:
    num_conv = []
    num_conv.append(3)
    num_conv.append(4)
    num_conv.append(6)
    num_conv.append(3)
  elif n == 101:
    num_conv = []
    num_conv.append(3)
    num_conv.append(4)
    num_conv.append(23)
    num_conv.append(3)
  elif n == 152:
    num_conv = []
    num_conv.append(3)
    num_conv.append(8)
    num_conv.append(36)
    num_conv.append(3)
  elif n == 34:
    num_conv = []
    num_conv.append(3)
    num_conv.append(4)
    num_conv.append(6)
    num_conv.append(3)
  
  layers = []
  
  with tf.variable_scope('conv1'):
    conv1 = conv_layer(inpt, [7, 7, 3, 64], 2, loc, tr_stat = _layer0_stat, \
        bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, state = "split")
    max1 = max_pool_3x3(conv1)
    layers.append(max1)
  
  for i in range (num_conv[0]):
    with tf.variable_scope('conv2_%d' % (i + 1)):
      conv2 = residual_block(layers[-1], 64, False, loc, tr_stat = _layer0_stat, \
          bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "near")
      layers.append(conv2)
  
  for i in range (num_conv[1]):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i + 1)):
      conv3 = residual_block(layers[-1], 128, down_sample, loc, tr_stat = _layer1_stat, \
          bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "far")
      layers.append(conv3)
  
  for i in range (num_conv[2]):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i + 1)):
      conv4 = residual_block(layers[-1], 256, down_sample, loc, tr_stat = _layer2_stat, \
          bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "far")
      layers.append(conv4)
  
  return layers[-1]


#SETUP
num_labels = 21
batch_size = 1

prep_img = tf.placeholder(tf.float32)
im_info = tf.placeholder(tf.float32, [3])
im_height = tf.placeholder(tf.int32)
im_width = tf.placeholder(tf.int32)
prep_img = tf.reshape(prep_img, [1, im_height, im_width, 3])
im_batch = prep_img
gt_box = tf.placeholder(tf.int64)
gt_boxbatch = tf.reshape(tf.pack(gt_box), [-1, 5])
height = tf.cast(tf.ceil(tf.cast(im_height, tf.float32)/16), tf.int32)
width = tf.cast(tf.ceil(tf.cast(im_width, tf.float32)/16), tf.int32)

#VGG_TRUNK
with tf.name_scope("trunk"):
  h_conv13 = resnet(im_batch, resnet_length, "trunk")

#RCNN
with tf.name_scope("rcnn"):
  r7 = residual_block(h_conv13, 512, False, "rcnn", tr_stat = _rcnn_stat, bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "near")
  r8 = residual_block(r7, 512, False, "rcnn", tr_stat = _rcnn_stat, bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "far")
  r9 = residual_block(r8, 512, False, "rcnn", tr_stat = _rcnn_stat, bn_tr_stat = _tr_bn, add_l2_stat = _add_l2, branch = "far")

#RPN
with tf.name_scope("rpn"):
  gate = tf.placeholder(tf.float32)
  h_rpn_input = (h_conv13 * (1-gate)) # gate is a residue of alternative-optimization
  W_rpn3 = weight_variable([3,3,2048,512], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  b_rpn3 = bias_variable([512], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  h_rpn3 = tf.nn.relu(conv2d(r9, W_rpn3) + b_rpn3)

  W_cls_score = weight_variable([1,1,512,18], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  b_cls_score = bias_variable([18], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  rpn_cls_score = (conv2d_nopad(h_rpn3, W_cls_score) + b_cls_score)

  W_bbox_pred = weight_variable_bbox([1,1,512,36], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  b_bbox_pred = bias_variable([36], "rpn", tr_stat = _rpn_stat, add_l2_stat = _add_l2)
  rpn_bbox_pred = (conv2d_nopad(h_rpn3, W_bbox_pred) + b_bbox_pred)

#RPN loss and accuracy calculation
rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [1, height, width, A * 4])
rpn_bbox_pred = tf.transpose(rpn_bbox_pred, [0, 3, 1, 2])
rpn_cls_score_t = tf.transpose(rpn_cls_score, [0, 3, 1, 2])
rpn_cls_score_reshape = tf.reshape(rpn_cls_score_t, [2, -1]) + 1e-20
rpn_cls_score_reshape = tf.transpose(rpn_cls_score_reshape, [1, 0])

rpn_labels_ind, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, rpn_size = \
  tf.py_func(anchor, [rpn_cls_score_t, gt_boxbatch, im_info], [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])

rpn_labels_ind = tf.reshape(tf.pack(rpn_labels_ind), [-1])
rpn_bbox_targets = tf.reshape(tf.pack(rpn_bbox_targets), [1,  A * 4, height, width])
rpn_bbox_inside_weights = tf.reshape(tf.pack(rpn_bbox_inside_weights), [1,  A * 4, height, width])
rpn_bbox_outside_weights = tf.reshape(tf.pack(rpn_bbox_outside_weights), [1,  A * 4, height, width])

rpn_cls_soft = tf.nn.softmax(rpn_cls_score_reshape) 
rpn_cls_score_x = tf.reshape(tf.gather(rpn_cls_score_reshape,tf.where(tf.not_equal(rpn_labels_ind,-1))),[-1,2])
rpn_label = tf.reshape(tf.gather(rpn_labels_ind, tf.where(tf.not_equal(rpn_labels_ind,-1))),[-1])
rpn_loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score_x, rpn_label))

unique_rpn_cls, o_cls, o_cls_ind = tf.py_func(cls_unique, \
    [rpn_cls_soft, rpn_labels_ind], [tf.float32, tf.float32, tf.float32])
unique_rpn_cls = tf.pack(unique_rpn_cls)

rpn_correct_prediction = tf.py_func(rpn_accuracy, [rpn_cls_soft, rpn_labels_ind], [tf.float32])
rpn_correct_prediction = tf.reshape(tf.pack(rpn_correct_prediction), [-1])
rpn_cls_accuracy = tf.reduce_mean(tf.cast(rpn_correct_prediction, tf.float32))

sigma1 = s1 * s1
smoothL1_sign = tf.cast(tf.less(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),1/sigma1),tf.float32)
rpn_loss_bbox = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rpn_bbox_outside_weights,tf.add( \
    tf.mul(tf.mul(tf.pow(tf.mul(rpn_bbox_inside_weights, \
        tf.sub(rpn_bbox_pred, rpn_bbox_targets)),2),0.5*sigma1), smoothL1_sign), \
    tf.mul(tf.sub(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),0.5/sigma1),\
        tf.abs(smoothL1_sign-1)))), reduction_indices=[1,2])),1)
rpn_loss_bbox_label = rpn_loss_bbox
zero_count, one_count = tf.py_func(bbox_counter, [rpn_labels_ind], [tf.float32, tf.float32])


#ROI PROPOSAL
rpn_cls_prob = rpn_cls_soft
rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob, [1, 18, height, width])

rpn_rois = tf.py_func(proposal, [rpn_cls_prob_reshape, rpn_bbox_pred, im_info], [tf.float32])
rpn_rois = tf.reshape(rpn_rois, [-1, 5])

rcnn_rois, rcnn_labels_ind, rcnn_bbox_targets, rcnn_bbox_inside_w, rcnn_bbox_outside_w = \
  tf.py_func(proposal_target, [rpn_rois, gt_boxbatch], [tf.int32, tf.int64, tf.float32, tf.float32, tf.float32])
rcnn_rois = tf.cast(tf.reshape(tf.pack(rcnn_rois), [-1, 5]), tf.float32)
rcnn_labels_ind = tf.reshape(tf.pack(rcnn_labels_ind), [-1])
rcnn_bbox_targets = tf.reshape(tf.pack(rcnn_bbox_targets), [-1, 2 * 4])
rcnn_bbox_inside_w = tf.reshape(tf.pack(rcnn_bbox_inside_w), [-1, 2 * 4])
rcnn_bbox_outside_w = tf.reshape(tf.pack(rcnn_bbox_outside_w), [-1, 2 * 4])

#FC
with tf.name_scope("fc"):
  
  W_end_base = weight_variable([1,1,2048,1024], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  b_end_base = bias_variable([1024], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  h_end_base = tf.nn.relu(conv2d_nopad(r9, W_end_base) + b_end_base)
  
  W_rfcn_cls = weight_variable([1,1,1024,1029], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  b_rfcn_cls = bias_variable([1029], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  h_rfcn_cls = (conv2d_nopad(h_end_base, W_rfcn_cls) + b_rfcn_cls)
  
  W_rfcn_bbox = weight_variable([1,1,1024,392], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  b_rfcn_bbox = bias_variable([392], "fc", tr_stat = _fc_stat, add_l2_stat = _add_l2)
  h_rfcn_bbox = (conv2d_nopad(h_end_base, W_rfcn_bbox) + b_rfcn_bbox)
  
  h_rfcn_cls = tf.transpose(h_rfcn_cls, [0, 3, 1, 2])
  [psroipooled_cls_rois, cls_channels] = psroi_pooling_op.psroi_pool(h_rfcn_cls, rcnn_rois, output_dim=21, group_size=7, spatial_scale=1.0/16)
  psroipooled_cls_rois = tf.transpose(psroipooled_cls_rois, [0, 2, 3, 1])
  end_cls = tf.reduce_mean(psroipooled_cls_rois, [1, 2])
  end_cls = tf.reshape(end_cls, [-1, 21])
  
  h_rfcn_bbox = tf.transpose(h_rfcn_bbox, [0, 3, 1, 2])
  [psroipooled_loc_rois, loc_channels] = psroi_pooling_op.psroi_pool(h_rfcn_bbox, rcnn_rois, output_dim=8, group_size=7, spatial_scale=1.0/16)
  psroipooled_loc_rois = tf.transpose(psroipooled_loc_rois, [0, 2, 3, 1])
  end_bbox = tf.reduce_mean(psroipooled_loc_rois, [1, 2])
  end_bbox = tf.reshape(end_bbox, [-1, 8])

#END_LOSS
end_cls_soft = tf.nn.softmax(end_cls) 
loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(end_cls, rcnn_labels_ind))
loss_cls_label = loss_cls

pred = tf.argmax(end_cls_soft, 1)
end_correct_prediction = tf.equal(pred, rcnn_labels_ind)
end_cls_accuracy = tf.reduce_mean(tf.cast(end_correct_prediction, tf.float32))

sigma2 = s2 * s2
smoothL1_sign_bbox = tf.cast(tf.less(tf.abs(tf.sub(end_bbox, rcnn_bbox_targets)),1/sigma2),tf.float32)
loss_bbox = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rcnn_bbox_outside_w,tf.add( \
    tf.mul(tf.mul(tf.pow(tf.mul(rcnn_bbox_inside_w, tf.sub(end_bbox, rcnn_bbox_targets))*1,2),0.5*sigma2), smoothL1_sign_bbox), \
    tf.mul(tf.sub(tf.abs(tf.sub(end_bbox, rcnn_bbox_targets)),0.5/sigma2),tf.abs(smoothL1_sign_bbox-1)))), reduction_indices=[1])),1)

total_loss = rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + tf.add_n(tf.get_collection('weight_losses_trunk')) + tf.add_n(tf.get_collection('weight_losses_rpn')) +\
     tf.add_n(tf.get_collection('weight_losses_rcnn')) +  tf.add_n(tf.get_collection('weight_losses_fc'))

#VARIABLES, OPTIMIZERS, AND LOSSES
trunk_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trunk") if "weights" in v.name]
trunk_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trunk") if "biases" in v.name]
rpn_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rpn") if "weights" in v.name]
rpn_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rpn") if "biases" in v.name]
rcnn_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rcnn") if "weights" in v.name]
rcnn_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rcnn") if "biases" in v.name]
fc_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc") if "weights" in v.name]
fc_biases = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc") if "biases" in v.name]

lr = tf.placeholder(tf.float32)
var_list1_w =  rcnn_weights + fc_weights + trunk_weights + rpn_weights
var_list1_b = rcnn_biases + fc_biases + trunk_biases + rpn_biases
opt1_w = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)
opt1_b = tf.train.MomentumOptimizer(lr*2, momentum=0.9, use_nesterov=True)
grads1 = tf.gradients(total_loss,  var_list1_w + var_list1_b)
grads1_w = grads1[:len(var_list1_w)]
grads1_b = grads1[len(var_list1_w):]
train_op1_w = opt1_w.apply_gradients(zip(grads1_w, var_list1_w))
train_op1_b = opt1_b.apply_gradients(zip(grads1_b, var_list1_b))
batchnorm_updates = tf.get_collection('update_ops')
batchnorm_updates_op = tf.group(*batchnorm_updates)
first_train_op = tf.group(train_op1_w, train_op1_b, batchnorm_updates_op)

anchor_fraction = one_count / (zero_count + one_count)

#TRAINING

trunk_vars = [v for v in tf.all_variables() if np.logical_and("trunk" in v.name, "Momentum" not in v.name)]
rpn_vars = [v for v in tf.all_variables() if np.logical_and("rpn" in v.name, "Momentum" not in v.name)]
rcnn_vars = [v for v in tf.all_variables() if np.logical_and("rcnn" in v.name, "Momentum" not in v.name)]
saver_all_trunkrcnn = tf.train.Saver(trunk_vars + rcnn_vars)
saver_all = tf.train.Saver()


run_size = 8000

accu_rpn_losscls = 0
accu_rpn_lossbbox = 0
accu_rcnn_losscls = 0
accu_rcnn_lossbbox = 0
accu_rpn_accuracy = 0
accu_rcnn_accuracy = 0
accu_anchor_fraction = 0
accu_anchor_count = 0

epoch = 40

init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init, feed_dict = {gate : [0.0]})
  saver_all_trunkrcnn.restore(sess, "./imagenet_resnet.ckpt")
  #saver_all_trunkrcnn.restore(sess, "./caltech_160k.ckpt")
  #saver_all.restore(sess,"./z7_300k.ckpt")
  #saver_all.restore(sess, "./z7_end_to_end_girschick.ckpt")
  for x in range(epoch):
      rand_id = np.random.permutation(np.arange(run_size)) 
      for i in range(run_size):
          if i % 10 == 0:
              print "epoch: " + str(x) + "  iter: " + str(x*run_size+i)
          
          #PROCESS IMAGE AND LABELS
          blobs = mnc.forward()
          img_train = blobs['data'].transpose(0,2,3,1)
          gt_train = blobs['gt_boxes']
          gt_train = gt_train.reshape(-1, 5).astype(np.int64)
          im_info_x = blobs['im_info'].reshape(-1).astype(np.float32)
          im_height_x = im_info_x[0].astype(np.int32)
          im_width_x = im_info_x[1].astype(np.int32)
          decay = np.floor(x/7)
          learn_rate = 0.001 * np.power(0.1,decay)
         
          #RUN TRAIN OP
          _, rpnlosscls, rpnlossbbox, losscls, lossbbox, rpnaccuracy, rcnnaccruacy, anchorfraction, anchorcount = \
              sess.run([first_train_op, rpn_loss_cls, rpn_loss_bbox, loss_cls, \
                  loss_bbox, rpn_cls_accuracy, end_cls_accuracy, anchor_fraction, o_cls], \
              feed_dict = {lr : learn_rate, gate : [0.0], prep_img : img_train, gt_box : gt_train, im_info : im_info_x, im_height : im_height_x, im_width : im_width_x})
          
          accu_rpn_losscls += rpnlosscls
          accu_rpn_lossbbox += rpnlossbbox
          accu_rcnn_losscls += losscls
          accu_rcnn_lossbbox += lossbbox
          accu_rpn_accuracy += rpnaccuracy
          accu_rcnn_accuracy += rcnnaccruacy
          accu_anchor_fraction += anchorfraction
          accu_anchor_count += anchorcount
          
          if np.logical_and(i % 100 == 0, (x+i) != 0):
              print ""
              print "training update:"
              print "average rpn_cls_loss:     " + str(accu_rpn_losscls/100)
              print "average rpn_bbox_loss:    " + str(accu_rpn_lossbbox/100)
              print "average end_cls_loss:     " + str(accu_rcnn_losscls/100)
              print "average end_bbox_loss:    " + str(accu_rcnn_lossbbox/100)
              print "average rpn accuracy:     " + str(accu_rpn_accuracy/100)
              print "average rcnn accuracy:    " + str(accu_rcnn_accuracy/100)
              u_anchors, indexes, pred_, lb_ = sess.run([unique_rpn_cls, o_cls_ind, pred, rcnn_labels_ind], \
                  feed_dict = {lr : learn_rate, gate : [0.0], prep_img : img_train, gt_box : gt_train, im_info : im_info_x, im_height : im_height_x, im_width : im_width_x})
              print "unique_anchors:           " + str(u_anchors/1)
              print "average correct anchors:  " + str(accu_anchor_count/100)
              print "average anchor_fraction:  " + str(accu_anchor_fraction/100)
              if np.logical_and(i == 100, x == 0):
                  old_indexes = indexes
              if i > 0:
                  if indexes.shape == old_indexes.shape:
                      change = np.equal(indexes, old_indexes)
                      test = np.sum(change == 0)
                      print "anchor change:            " + str(test)
                  else:
                      print "there's change"
                      print ""
                  old_indexes = indexes
              print "predictions:"
              print pred_
              print "gt:"
              print lb_
              print ""
              
              accu_rpn_losscls = 0
              accu_rpn_lossbbox = 0
              accu_rcnn_losscls = 0
              accu_rcnn_lossbbox = 0
              accu_rpn_accuracy = 0
              accu_rcnn_accuracy = 0
              accu_anchor_fraction = 0
              accu_anchor_count = 0
          
          if i % 2000 == 0:
              save_path = saver_all.save(sess, "./rfcn_end_to_end.ckpt")
              print ""
              print "model saved"
              print ""
           
  sess.close()
