import numpy as np
import math, gzip, cPickle
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from scipy.sparse.linalg import  cg

class DataHolder:
    def __init__(self, dataset = 'MNIST', lam1 =0, lam2 = 0):
        self.lam1 = lam1
        self.lam2 = lam2
        self.load_dataset(dataset)


    def load_dataset(self, dataset):
        if dataset == 'MNIST':
            print '-----------------------------------------------'
            print 'Loading MNIST 4/9 data...'
            print '-----------------------------------------------\n'
            self.load_mnist_49()
            self.data_dim = self.train_set[0][0].size
            self.num_train_examples = self.train_set[0].shape[0]
            self.num_test_examples = self.test_set[0].shape[0]

        if dataset == 'covtype':
            print '-----------------------------------------------'
            print 'Loading covtype 3/6 data...'
            print '-----------------------------------------------\n'
            self.load_covtype_36()
            self.data_dim = self.train_set[0][0].size
            self.num_train_examples = self.train_set[0].shape[0]

    ## This is to load the 49 mnist
    def load_mnist_49(self):
        f = open('../data/mnist49data', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]
        self.valid_set = [normalize(valid_set[0], axis=1, norm='l2'), valid_set[1]]
        self.test_set = [normalize(test_set[0], axis=1, norm='l2'), test_set[1]]

    def load_covtype_36(self):
        f = open('/Users/hsliu/Desktop/charm/data/covtype36', 'rb')
        train_set = cPickle.load(f)
        f.close()
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]

    ### The following functions implement the logistic function oracles

    def fetch_correct_datamode(self, mode='TRAIN'):
        if mode == 'TRAIN':
            return self.train_set
        elif mode == 'VALIDATE':
            return self.validate_set
        elif mode == 'TEST':
            return self.test_set
        else:
            raise ValueError('Wrong mode value provided')


    ## function values for sample
    def logistic_indiv_func(self, data_index, model, mode ='TRAIN'):
        data_set = self.fetch_correct_datamode(mode)
        v = -1.0*data_set[1][data_index] * np.dot(model, data_set[0][data_index])
        return np.log(np.exp(v) + 1)

    ## function gradient for sample
    def logistic_indiv_grad(self, data_index, model):
        data_set = self.train_set
        v = -1.0 * data_set[1][data_index] * np.dot(model, data_set[0][data_index])
        return -1 * data_set[1][data_index] * data_set[0][data_index] * (np.exp(v) / (1 + np.exp(v)))


    def logistic_indiv_grad_coeff(self, data_index, model):
        data_set = self.train_set
        v = -1.0 * data_set[1][data_index] * np.dot(model, data_set[0][data_index])
        return -1 * data_set[1][data_index] * (np.exp(v) / (1 + np.exp(v)))

    ##
    def logistic_batch_func(self, data_batch, model, mode='TRAIN'):
        func_val = 0.0
        for data_indiv in data_batch:
            func_val += self.logistic_indiv_func(data_indiv, model, mode)
        avg_func_val = func_val / len(data_batch)
        return avg_func_val + self.lam2/2.0 * np.dot(model, model) + self.lam1*LA.norm(model,1)

    def logistic_batch_grad(self, data_batch, model):
        batch_grad = np.zeros(self.data_dim)
        for data_indiv in data_batch:
            batch_grad += self.logistic_indiv_grad(data_indiv, model)
        avg_batch_grad = batch_grad / len(data_batch)
        return avg_batch_grad + 2 * self.lam2 * model

    def test_error(self, model):
        func_val = 0.0
        data_batch = range(0, self.num_test_examples)
        for data_indiv in data_batch:
            func_val += self.logistic_indiv_func(data_indiv, model, 'TEST')
        avg_func_val = func_val / len(data_batch)
        return avg_func_val



























