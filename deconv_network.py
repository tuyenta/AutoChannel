import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

import ipdb

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


def conv2d(x, nfilters=10, kernel_size=(1,3,1), strides=[1, 1, 1, 1], padding='SAME', activation=None, name='l1', seed=None):
  z = tf.to_int32(tf.shape(x)[2])
  z2 = x.get_shape().as_list()
  kernel_size = list(kernel_size)
  kernel_size[2] = z2[3]

  with tf.variable_scope(name):
    W = tf.get_variable("weights", shape=list(kernel_size)+[nfilters],  # tf.shape_n(x)[0]
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed))
            
    b_conv = bias_variable_n([nfilters], name)
  return tf.nn.relu(tf.nn.conv2d(x, W, strides=strides, padding=padding) + b_conv) 

def conv2d_transpose(x, output_shape, nfilters=10, kernel_size=(1,3,1), strides=[1, 1, 1, 1], padding='SAME', activation=None,  seed=None):
  with tf.name_scope('conv_transpose'):
    z = tf.to_int32(tf.shape(x)[2])
    z2 = x.get_shape().as_list()
    kernel_size = list(kernel_size)
    kernel_size[2] = z2[3]

#    with tf.variable_scope(name):
    W = tf.get_variable("weights", shape=list(kernel_size)+[nfilters],  
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed))
            
    b_conv = bias_variable_n([output_shape[2]], name)
    return tf.nn.relu(tf.nn.conv2d_transpose(x, W, strides=strides, padding=padding) + b_conv) 



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



def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
    N = X.shape[0]
    NoSample = X.shape[1]
    print("N: %d" %N)
    out = np.zeros((N, NoSample, 1, 1), dtype=np.float32)
    for i in range(N-1):
        for j in range(NoSample-1):
            out[i, j, 0, 0] = X[i, j]
    return out

def error_rate(p, t):
  return np.mean(p != t)


class TFLogistic:
  def __init__(self, savefile, M=None, K=None, batch_sz=None):
    self.Summary=None
    self.savefile = savefile
    self.learning_rate = 0.0001
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

    self.X = tf.placeholder(tf.float32, shape=(None,M,1,1), name='X')
    self.T = tf.placeholder(tf.float32, shape=(None,K), name='T')

    self.W1, self.b1 = self.weightInital([5, 1, 1, 64], 1)
    self.W2, self.b2 = self.weightInital([3, 1, 64, 64], 1)


    self.W3, self.b3 = self.weightInital([3, 1, 64, 128], 1)
    self.W4, self.b4 = self.weightInital([3, 1, 128, 128], 1)

    self.W5, self.b5  = self.weightInital([3,1,128,256],1)
    self.W6, self.b6  = self.weightInital([3,1,256,256],1)

    self.W7, self.b7  = self.weightInital([3,1,256,256],1)

    self.W8, self.b8  = self.weightInital([3,1,256,512],1)
    self.W9, self.b9  = self.weightInital([3,1,512,1024],1)
    
    self.WD4, self.bD4 = self.weightInital([3, 1, 128, 128], 2)
    self.WD3, self.bD3 = self.weightInital([3, 1, 64, 128], 2)
    self.WD2, self.bD2 = self.weightInital([3, 1, 64, 64], 2)
    self.WD1, self.bD1 = self.weightInital([5, 1, 1, 64], 2)        
    
    self.saver = tf.train.Saver({\
        'W1': self.W1, 'b1': self.b1,\
        'W2': self.W2, 'b2': self.b2,\
        'W3': self.W3, 'b3': self.b3,\
        'W4': self.W4, 'b4': self.b4,\
        'W5': self.W5, 'b5': self.b5,\
        'W6': self.W6, 'b6': self.b6,\
        'W7': self.W7, 'b7': self.b7,\
        'W8': self.W8, 'b8': self.b8,\
        'W9': self.W9, 'b9': self.b9})

    E1, index1, shape_in1 = convpool_argmax(self.X, self.W1, self.b1, ksize=[1, 2, 1, 1], name='l1')
    E2, index2, shape_in2 = convpool_argmax(E1, self.W2, self.b2, ksize=[1, 2, 1, 1], name='l2')
    E3, index3, shape_in3 = convpool_argmax(E2, self.W3, self.b3, ksize=[1, 2, 1, 1], name='l3')

    D3 = convunpool(E3, index3, shape_in3, self.WD3, self.bD3, ksize=[1, 2, 1, 1])
    D2 = convunpool(D3, index2, shape_in2, self.WD2, self.bD2, ksize=[1, 2, 1, 1])
    D1 = convunpool(D2, index1, shape_in1, self.WD1, self.bD1, ksize=[1, 2, 1, 1])

    Z1 = convpool(self.X, self.W1, self.b1)
    Z2 = convpool(Z1, self.W2, self.b2)
    Z3 = convpool(Z2, self.W3, self.b3)
    Z4 = convpool(Z3, self.W4, self.b4)
    Z5 = convpool(Z4, self.W5, self.b5)
    Z6 = convpool(Z5, self.W6, self.b6)
    Z7 = convpool(Z6, self.W7, self.b7)
    Z8 = convpool(Z7, self.W8, self.b8)
    Z9 = convpool(Z8, self.W9, self.b9)

#    Yish = tf.reshape(Z9,[-1,1024,2])
    Yish = tf.reshape(D1,[-1,1024])

    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=self.T))
    
    tf.summary.scalar('loss', cost)
    self.Summary = tf.summary.merge_all()

#    self.predict_op = tf.argmax(Yish, 2)
    self.predict_op = tf.cast(tf.round(Yish), tf.int32)
    return cost


  def fit(self, X, Y, Xtest, Ytest):
    N, M, oo, ox = X.shape
    K = Y.shape[1]
#    Ytest_all = ind2y(Ytest)
    Ytest_all = Ytest

    # hyperparams
    max_iter = 1
    print_period = 10
    batch_sz = 500
    n_batches = int(N / batch_sz)

    cost = self.build(M, K)
#    train_op = tf.train.RMSPropOptimizer(0.00005, decay=0.99, momentum=0.9).minimize(cost)
    train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

    t0 = datetime.now()
    costs = []
    init = tf.initialize_all_variables()
    
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
        device_count = {'GPU': 1}   # number of devices available
    )
    
    
    with tf.Session(config=config) as session:
        session.run(init)
        writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph()) ## Tensorboard
        X, Y = shuffle(X, Y, random_state = 65)
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz),]
                #Xbatch, Ybatch = shuffle(Xbatch, Ybatch, random_state = 65)
                #YbatchInd = y2indicator3D(Ybatch)
                #print YbatchInd.shape
                
                

                if len(Xbatch) == batch_sz:
                    _, summary = session.run([train_op, self.Summary], feed_dict={self.X: Xbatch, self.T: Ybatch})
                    writer.add_summary(summary)
                    if j % print_period == 0:
                        
                        test_cost = 0
                        prediction = np.zeros((len(Xtest),K))
                        #print prediction.shape
                        for k in range(len(Xtest) // batch_sz):
                            Xtestbatch = Xtest[k*batch_sz:(k*batch_sz + batch_sz),]
                            Ytestbatch = Ytest[k*batch_sz:(k*batch_sz + batch_sz),]
                            #YtestbatchInd = y2indicator3D(Ytestbatch)

                            test_cost += session.run(cost, feed_dict={self.X: Xtestbatch, self.T: Ytestbatch})
                            prediction[k*batch_sz:(k*batch_sz + batch_sz),] = session.run(
                                self.predict_op, feed_dict={self.X: Xtestbatch})
                            #print prediction.shape



                        # print Ytest.shape
                        # print prediction.shape
                        err = error_rate(prediction, Ytest_all)
                        #print "Predict type:", prediction
                        #print "Real type:", Ytest_all
                        print("At epoch=%d, batch=%d, cost value / error: %.3f / %.3f" % (i, j, test_cost, err))
                        costs.append(test_cost)


        # save the model
        save_path = self.saver.save(session, self.savefile)
        print("model save in file: %s" %save_path)

    # save dimensions for later
    self.M = M
    self.K = K
    #self.batch_sz=batch_sz
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(costs)
    plt.show()


  def predict(self, X):
    
    with tf.Session() as session:
      # restore the model
      self.saver.restore(session, self.savefile)
      P = session.run(self.predict_op, feed_dict={self.X: X})
    return P

  def predict00(self, X):
    with tf.Session() as session:
      # restore the model
      self.saver.restore(session, self.savefile)
      t00 = datetime.now()
      P = session.run(self.predict_op, feed_dict={self.X: X})
      print("Elapsed time:", (datetime.now() - t00))
    return P


  def score(self, X, Y):
    batch_sz=500;

    if len(X) <= batch_sz:
        Ypredict = self.predict(X)
        Ytrue = Y
    else:
        Ytrue=Y[:(len(X) // batch_sz)*batch_sz]
        Ypredict = []
        for k in xrange(len(X) // batch_sz):
            Xbatch = X[k*batch_sz:(k+1)*batch_sz,]
            Ypredict[k*batch_sz:(k+1)*batch_sz] = self.predict(Xbatch)

    return 1 - error_rate(Ypredict, Ytrue)

  def save(self, filename):
    j = {
      'M': self.M,
      'K': self.K,
      'model': self.savefile
    }
    with open(filename, 'w') as f:
      json.dump(j, f)

  @staticmethod
  def load(filename):
    with open(filename) as f:
      j = json.load(f)
    return TFLogistic(j['model'], j['M'], j['K'])


def main():

#    TT = loadmat('RRTimeSeriesLongmat.mat') 
    TT = loadmat('../data/RRTimeSeriesLong1024_100.mat') 
    X = rearrange(TT['P'])
    Y = TT['label']
    del TT
    X, Y = shuffle(X, Y, random_state =39)

    Xtrain = X[:-10300,]
    Ytrain = Y[:-10300]
    Xtest  = X[-10300:-300,]
    Ytest  = Y[-10300:-300]


    print("Trainning length: %d , Testing length: %d" %(len(Ytrain), len(Ytest)))

    #print Ytrain.shape


    Ytrain_ind = y2indicator3D(Ytrain)
    Ytest_ind  = y2indicator3D(Ytest)

    # print Ytest_ind.shape


    batch_sz = 500

    trainload = 1

    if trainload ==1: 
        model = TFLogistic("segmentation1024.model")
#        model.fit(Xtrain, Ytrain_ind,Xtest, Ytest_ind)
        model.fit(Xtrain, Ytrain,Xtest, Ytest)
        # test out restoring the model via the predict function
        print("final train accuracy:", model.score(Xtrain, Ytrain))
        print("final test accuracy:", model.score(Xtest, Ytest))
        # save the model
        model.save("segmentation1024.json")

    # # load and score again
    model = TFLogistic.load("segmentation1024.json")

    # print "%s" %model.savefile
    print("final train accuracy (after reload):", model.score(Xtrain, Ytrain))
    print("final test accuracy (after reload):", model.score(Xtest, Ytest))
    print("one shot:", (model.predict00([Xtrain[10,]])))

if __name__ == '__main__':
    main()
