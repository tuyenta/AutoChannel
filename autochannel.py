import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

def ind2y(ind):
    N = ind.shape[0]
    NoSample = ind.shape[1]
    dim = ind.shape[2]
    y = np.zeros((N,NoSample))
    for i in xrange(N):
        for j in xrange(NoSample):
            for k in xrange(dim): 
                if ind[i,j,k]==1:
                    y[i,j] = k
    return y

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 2))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind

def y2indicator3D(y):
    L,NoSample = y.shape
    maxInd = y.max()
    ind = np.zeros((L,NoSample, maxInd + 1))
    for i in xrange(L):
        for j in xrange(NoSample):
            ind[i, j , y[i,j]] = 1
    return ind


def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)

def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
    N = X.shape[0]
    NoSample = X.shape[1]
    print "N: %d" %N
    out = np.zeros((N, NoSample, 1, 1), dtype=np.float32)
    for i in xrange(N-1):
        for j in xrange(NoSample-1):
            out[i, j, 0, 0] = X[i, j]
    return out

def error_rate(p, t):
  return np.mean(p != t)


class TFLogistic:
  def __init__(self, savefile, M=None, K=None, batch_sz=None):
    self.savefile = savefile
    if M and K:
      # we can define some parts in the model to be able to make predictions
      self.build(M, K)
  
  def weightInital(self, W_shape,cnn):
    if cnn:
        poolsz = (2,1)
        W_init = init_filter(W_shape,poolsz)
        b_init = np.zeros(W_shape[-1], dtype=np.float32)
    else:
        W_init = np.random.randn(W_shape[0],W_shape[1]) / np.sqrt(W_shape[0] + W_shape[1])
        b_init = np.zeros(W_shape[1], dtype=np.float32)

    return tf.Variable(W_init.astype(np.float32)), tf.Variable(b_init.astype(np.float32))
        



  def build(self, M, K):

    self.X = tf.placeholder(tf.float32, shape=(None,M,1,1), name='X')
    self.T = tf.placeholder(tf.float32, shape=(None,K,2), name='T')

    self.W1, self.b1 = self.weightInital([5, 1, 1, 64], 1)
    self.W2, self.b2 = self.weightInital([3, 1, 64, 64], 1)


    self.W3, self.b3 = self.weightInital([3, 1, 64, 128], 1)
    self.W4, self.b4 = self.weightInital([3, 1, 128, 128], 1)

    self.W5, self.b5  = self.weightInital([3,1,128,256],1)
    self.W6, self.b6  = self.weightInital([3,1,256,256],1)

    self.W7, self.b7  = self.weightInital([3,1,256,256],1)

    self.W8, self.b8  = self.weightInital([3,1,256,512],1)
    self.W9, self.b9  = self.weightInital([3,1,512,1024],1)
    
    
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

    Z1 = convpool(self.X, self.W1, self.b1)
    Z2 = convpool(Z1, self.W2, self.b2)
    Z3 = convpool(Z2, self.W3, self.b3)
    Z4 = convpool(Z3, self.W4, self.b4)
    Z5 = convpool(Z4, self.W5, self.b5)
    Z6 = convpool(Z5, self.W6, self.b6)
    Z7 = convpool(Z6, self.W7, self.b7)
    Z8 = convpool(Z7, self.W8, self.b8)
    Z9 = convpool(Z8, self.W9, self.b9)
    Yish = tf.reshape(Z9,[-1,1024,2])

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits (Yish, self.T))
    
    self.predict_op = tf.argmax(Yish, 2)
    return cost


  def fit(self, X, Y, Xtest, Ytest):
    N, M, oo, ox = X.shape
    K = Y.shape[1]
    Ytest_all = ind2y(Ytest)

    # hyperparams
    max_iter = 30
    print_period = 10
    batch_sz = 500
    n_batches = N / batch_sz

    cost = self.build(M, K)
    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    t0 = datetime.now()
    costs = []
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        X, Y = shuffle(X, Y, random_state = 65)
        for i in xrange(max_iter):
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz),]
                #Xbatch, Ybatch = shuffle(Xbatch, Ybatch, random_state = 65)
                #YbatchInd = y2indicator3D(Ybatch)
                #print YbatchInd.shape

                if len(Xbatch) == batch_sz:
                    session.run(train_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    if j % print_period == 0:
                        test_cost = 0
                        prediction = np.zeros((len(Xtest),K))
                        #print prediction.shape
                        for k in xrange(len(Xtest) / batch_sz):
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
                        print "At epoch=%d, batch=%d, cost value / error: %.3f / %.3f" % (i, j, test_cost, err)
                        costs.append(test_cost)


        # save the model
        save_path = self.saver.save(session, self.savefile)
        print("model save in file: %s" %save_path)

    # save dimensions for later
    self.M = M
    self.K = K
    #self.batch_sz=batch_sz
    print "Elapsed time:", (datetime.now() - t0)
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
      print "Elapsed time:", (datetime.now() - t00)
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

    TT = loadmat('RRTimeSeriesLongmat.mat') 
    X = rearrange(TT['P'])
    Y = TT['label']
    del TT
    X, Y = shuffle(X, Y, random_state =39)

    Xtrain = X[:-10300,]
    Ytrain = Y[:-10300]
    Xtest  = X[-10300:-300,]
    Ytest  = Y[-10300:-300]


    print "Trainning length: %d , Testing length: %d" %(len(Ytrain), len(Ytest))

    #print Ytrain.shape


    Ytrain_ind = y2indicator3D(Ytrain)
    Ytest_ind  = y2indicator3D(Ytest)

    # print Ytest_ind.shape


    batch_sz = 500

    trainload = 1;

    if trainload ==1: 
        model = TFLogistic("NhanTF_Segmentation1024.model")
        model.fit(Xtrain, Ytrain_ind,Xtest, Ytest_ind)
        # test out restoring the model via the predict function
        print "final train accuracy:", model.score(Xtrain, Ytrain)
        print "final test accuracy:", model.score(Xtest, Ytest)
        # save the model
        model.save("NhanTF_Segmentation1024.json")

    # # load and score again
    model = TFLogistic.load("NhanTF_Segmentation1024.json")

    # print "%s" %model.savefile
    print "final train accuracy (after reload):", model.score(Xtrain, Ytrain)
    print "final test accuracy (after reload):", model.score(Xtest, Ytest)
    print "one shot:", (model.predict00([Xtrain[10,]]))

if __name__ == '__main__':
    main()