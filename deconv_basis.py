import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime
import ipdb

def ind2y(ind):
    N = ind.shape[0]
    NoSample = ind.shape[1]
    dim = ind.shape[2]
    y = np.zeros((N,NoSample))
    for i in range(N):
        for j in range(NoSample):
            for k in range(dim): 
                if ind[i,j,k]==1:
                    y[i,j] = k
    return y

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 2))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def y2indicator3D(y):
    L,NoSample = y.shape
    maxInd = y.max()
    ind = np.zeros((L,NoSample, maxInd + 1))
    for i in range(L):
        for j in range(NoSample):
            ind[i, j , y[i,j]] = 1
    return ind


def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
  """
     Unpooling layer after max_pool_with_argmax.
     Args:
         pool:   max pooled output tensor
         ind:      argmax indices
         ksize:     ksize is the same as for the pool
     Return:
         unpool:    unpooling tensor
     source: https://github.com/tensorflow/tensorflow/issues/2169
  """
  with tf.variable_scope(scope):

    input_shape =  tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

    flat_input_size = tf.cumprod(input_shape)[-1]
    flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

    pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                      shape=tf.stack([input_shape[0], 1, 1, 1]))
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, tf.stack([flat_input_size, 1]))
    ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
    ind_ = tf.concat([b, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, tf.stack(output_shape))
    return ret    



def max_pool(inp, k=2):
    return tf.nn.max_pool_with_argmax_and_mask(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def max_unpool(inp, argmax, argmax_mask, k=2):
    return tf.nn.max_unpool(inp, argmax, argmax_mask, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def max_pool(inp, k=2):
    return tf.nn.max_pool_with_argmax_and_mask(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def max_unpool(inp, argmax, argmax_mask, k=2):
    return tf.nn.max_unpool(inp, argmax, argmax_mask, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def my_initializer(init_type='he_normal', seed=None):

  if init_type == 'glorot_normal':
    scale = 1.0
    mode = 'FAN_AVG'
    uniform = False  
  elif init_type == 'glorot_uniform':
    scale = 1.0
    mode = 'FAN_AVG'
    uniform = True    
  elif init_type == 'he_uniform':
    scale = 2.0
    mode = 'FAN_IN'
    uniform = True
  elif init_type == 'he_normal':
    scale = 2.0 / 1.3
    mode = 'FAN_IN'
    uniform = False
  else: # 'truncated normal'
    scale = 1.0 / 1.3
    mode = 'FAN_IN'
    uniform = False
   
  weight_init = tf.contrib.layers.variance_scaling_initializer(factor=scale, mode=mode, uniform=uniform, seed=seed)  
  return weight_init


from tensorflow.contrib.layers.python.layers import regularizers
def dense(x, size, activation=None, init_type='he_normal', bias_init=tf.zeros_initializer, reg_l2=0.0, seed=None, name='dense'):
  size = int(size) # size - numpy.int64 has problem
  in_shp = x.get_shape().as_list()
  in_shp = float(in_shp[-1])
#  print(in_shp, np.sqrt(in_shp))
  
  return tf.contrib.layers.fully_connected(x, size, 
                                           activation_fn=activation,
                                           weights_initializer=my_initializer(init_type, seed=seed),
                                           weights_regularizer=regularizers.l2_regularizer(reg_l2),
                                           biases_initializer=bias_init(),
                                           scope=name)

def flatten(x):
#  dim = tf.reduce_prod(x.get_shape().as_list()[1:])
  dim = np.prod(np.asarray(x.get_shape().as_list())[1:])
  return tf.reshape(x, [-1, dim])


def conv2d(x, nfilters=10, kernel_size=(1,3,1), strides=[1, 1, 1, 1], padding='SAME', activation=None, name='l1', seed=None):
  z = tf.to_int32(tf.shape(x)[2])
  z2 = x.get_shape().as_list()
  kernel_size = list(kernel_size)
  kernel_size[2] = z2[3]

  with tf.variable_scope(name+'_conv2d'):
    W = tf.get_variable("weights", shape=list(kernel_size)+[nfilters],  # tf.shape_n(x)[0]
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed))
            
    b_conv = tf.Variable(tf.constant(0.0, shape=[nfilters]))
  return tf.nn.relu(tf.nn.conv2d(x, W, strides=strides, padding=padding) + b_conv) 

def conv2d_transpose(x, nfilters=10, kernel_size=(1,3,1), strides=[1, 1, 1, 1], padding='SAME', activation=None,  name='l1', seed=None):
  with tf.name_scope('conv_transpose'):
    shp = x.get_shape().as_list()
    shp = tf.shape(x)
    input_depth = shp[3]
    input_size_h = shp[1]
    input_size_w = shp[2]
    
    filter_size_h = kernel_size[0]
    filter_size_w = kernel_size[1]
    in_channel_size=kernel_size[2]

    stride_h = strides[1]
    stride_w = strides[2]
    if padding == 'VALID':
      output_size_h = (input_size_h - 1)*stride_h + filter_size_h
      output_size_w = (input_size_w - 1)*stride_w + filter_size_w
    elif padding == 'SAME':
      output_size_h = (input_size_h - 1)*stride_h + 1
      output_size_w = (input_size_w - 1)*stride_w + 1
    else:
      raise ValueError("unknown padding")
    output_shape = tf.stack([shp[0], 
                            output_size_h, output_size_w, 
                            nfilters])
        
#    w_shape = tf.concat([tf.constant(list(kernel_size)), tf.reshape(input_depth, [1,])], 0)
#    tf.constant(list(kernel_size)) + [input_depth]                    
    with tf.variable_scope(name+'_convT'):
      W = tf.get_variable("weights", shape=list(kernel_size[0:-1])+[nfilters]+[in_channel_size],  
              initializer=tf.contrib.layers.xavier_initializer_conv2d(seed))
            
      b_conv = tf.Variable(tf.constant(0.0, shape=[nfilters]))
      return tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding=padding) + b_conv ) 

def conv2d_transpose2(x, nfilters=10, kernel_size=[3,1,1], strides=[1, 1], padding='same', activation=tf.nn.relu, init_type='glorot_uniform',  seed=None):
  with tf.name_scope('conv_transpose'):

    return tf.layers.conv2d_transpose(x, nfilters, kernel_size=kernel_size, strides=strides, padding=padding,
    kernel_initializer=my_initializer(init_type, seed=seed)) 


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)

def convpool_argmax(X, W, b, ksize=[1, 2, 2, 1], name='l1'):
  # just assume pool size is (2,2) because we need to augment it with 1s
  with tf.name_scope('conv_argmax_'+name):
    shape = X.get_shape().as_list()
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out, index = tf.nn.max_pool_with_argmax(conv_out, ksize=ksize, strides=ksize, padding='SAME')
    return [tf.nn.relu(pool_out), index, tf.shape(X)]


def convunpool(X, index, output_shape, W, b, ksize=[1, 2, 2, 1]):
  with tf.name_scope('conv_unpool'):
    unpooled = unpool(X, index, ksize=ksize)
    conv_unpooled = tf.nn.conv2d_transpose(unpooled, W, output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')
    conv_unpooled = tf.nn.bias_add(conv_unpooled, b)
    conv_unpooled = tf.nn.relu(conv_unpooled)
    return conv_unpooled
  



class TFLogistic:
  def __init__(self, savefile, M=None, K=None, batch_sz=None):
    self.savefile = savefile
    if M and K:
      # we can define some parts in the model to be able to make predictions
      self.build(M, K)
  
  def weightInital(self, W_shape,cnn):

    if cnn==1:
      poolsz = (2,1)
      W_init = init_filter(W_shape,poolsz)
      b_init = np.zeros(W_shape[-1], dtype=np.float32)
    elif cnn==2: # deconv
      poolsz = (2,1)
      W_init = init_filter(W_shape,poolsz)
      b_init = np.zeros(W_shape[-2], dtype=np.float32)    
    else:
      W_init = np.random.randn(W_shape[0],W_shape[1]) / np.sqrt(W_shape[0] + W_shape[1])
      b_init = np.zeros(W_shape[1], dtype=np.float32)

    return tf.Variable(W_init.astype(np.float32)), tf.Variable(b_init.astype(np.float32))
        



  def build(self, M, K):
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
        device_count = {'GPU': 1}   # number of devices available
    )
  
    batch_size = 8
    x = (np.random.randint(0,4, (batch_size, M,1,1))).astype(np.float32)
    Xbatch = x

    self.X = tf.placeholder(tf.float32, shape=(None,M,1,1), name='X')
#    self.T = tf.placeholder(tf.float32, shape=(None,K,2), name='T')


    
    l1 = [5, 1, 1, 64, 1]
    rseed = 2222
    ksize1 = [1, 2, 1, 1]
    FC_size = 1024
    
            
    E1_conv1 = conv2d(self.X, nfilters=64, kernel_size=(5,1,1), strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu, seed=rseed)
    E1, index1 = tf.nn.max_pool_with_argmax(E1_conv1, ksize=ksize1, strides=ksize1, padding='SAME')    
    shape_in1 = tf.shape(E1)

    E2 = conv2d(self.X, nfilters=256, kernel_size=(5,1,64), strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu, name='l2', seed=rseed)
    
    
    D1_unpool = unpool(E1, index1, ksize=[1, 2, 1, 1])


    D1 = conv2d_transpose(D1_unpool, nfilters=2,  kernel_size=(5,2,64), strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu,  seed=rseed)


    
    predict_op = tf.cast(tf.round(D1), tf.int32) #tf.where(D1>0.5, tf.constant(1), tf.constant(0))
    
    stuff = tf.reduce_sum(predict_op)
    
    
    tf.summary.scalar('loss', stuff)
    self.Summary = tf.summary.merge_all()

    
    print(self.X.get_shape())

    
    n_batches=1
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph()) ## Tensorboard
      
#      ipdb.set_trace(context=3)
            
      for j in range(n_batches):

        d1, summary = sess.run([D1, self.Summary], feed_dict={self.X: Xbatch})
        writer.add_summary(summary)
        
        pp = sess.run(D1, feed_dict={self.X: Xbatch})
        sess.run(D1, feed_dict={self.X: Xbatch}).shape
        
#        print('outshape', d1)
        
        ipdb.set_trace()
    


def main():

  model = TFLogistic("segmentation1024.model")
  model.build(256, 1)
        # test out restoring the model via the predict function

if __name__ == '__main__':
    main()    