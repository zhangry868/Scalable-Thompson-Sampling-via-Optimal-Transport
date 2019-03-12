import math

import numpy as np
import pdb
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano.tensor as T
import theano
from scipy.spatial.distance import pdist, squareform, cdist
import random

class svgd_bayesnn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, context_dim, num_actions, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1):
        self.n_hidden = n_hidden
        self.d = context_dim # number of data, dimension
        self.num_class = num_actions # number of actions
        self.M = M # number of particles
        
        
        self.num_vars = self.d * n_hidden + n_hidden * (1 + self.num_class) + self.num_class + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*num_class; b2 = num_class; 2 variances
        self.theta = np.zeros([self.M, self.num_vars])  # particles, will be initialized later
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.matrix('y') # labels
        mask = T.matrix('mask') # labels
        
        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        w_2 = T.matrix('w_2') # weights between hidden layer and output layer
        b_2 = T.vector('b_2') # bias of output
        
        N = T.scalar('N') # number of observations
        
        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')
        
        ###
        # prediction = T.nnet.nnet.softmax(T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2)
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2
        
        ''' define the log posterior distribution '''
        # log_lik_data = T.sum(T.sum(y * mask * T.log(prediction))) + 1e-3 * log_gamma

        log_lik_data = - T.sum(T.power(prediction - y, 2) * mask) + 1e-3 * log_gamma

        # log_prior_w = -0.5 * (self.num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + (b_2**2).sum())  \
        #                + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda
        
        priorprec = T.log(b0/a0)
        log_prior_w = -0.5 * (self.num_vars-2) * (T.log(2*np.pi)-priorprec) - (T.exp(priorprec)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + (b_2**2).sum())  \
                        + 1e-3* log_lambda

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])
        
        # automatic gradient
        self.logp_gradient = theano.function(
             inputs = [X, y, mask, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda],
             allow_input_downcast=True
        )
        
        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction, allow_input_downcast=True)

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)

        print 'Successfully initialized the models!'

    '''
        Training with SVGD
    '''
    def train(self, X_train, y_train, weights, batch_size = 512, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        N0 = X_train.shape[0]  # number of observations
        
        # for i in range(self.M):
        #     w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
        #     # # use better initialization for gamma
            # ridx = np.random.choice(range(X_train.shape[0]), \
            #                                np.min([X_train.shape[0], 1000]), replace = False)
            # y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2)
            # loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            # self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)
        
        grad_theta = np.zeros([self.M, self.num_vars])  # gradient 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        iter = 0
        batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
            dw1, db1, dw2, db2, dloggamma, dloglambda = self.logp_gradient(X_train[batch,:], y_train[batch], weights[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
            grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)
            
        # calculating the kernel matrix
        if(self.M > 1):
            kxy, dxkxy = self.svgd_kernel(h=-1)  
            grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)
        ########### Stop Here ###############

        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
        else:
            historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
        adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
        self.theta = self.theta + master_stepsize * adj_grad 
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, self.num_class)
        b2 = np.zeros((self.num_class,))
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)
    
    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    
    '''
        Pack all parameters in our model
    '''    
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2.flatten(), b2, [loggamma],[loglambda]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2 = np.reshape(w[:self.n_hidden*self.num_class],  [self.n_hidden, self.num_class])
        b2 = w[self.n_hidden*self.num_class:(self.n_hidden+1)*self.num_class]

        w = w[(self.n_hidden+1)*self.num_class:]
        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]
        
        return (w1, b1, w2, b2, loggamma, loglambda)

    def predict(self, X_test):
        pred_y_test = np.zeros([X_test.shape[0]])
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        ########### Random sample from the posterior distributions ##############
        i = np.random.randint(self.M)

        w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
        # pdb.set_trace()
        pred_y_test = (self.nn_predict(X_test, w1, b1, w2, b2))
        pred = pred_y_test

        return pred


class ensemble_nn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, context_dim, num_actions, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1):
        self.n_hidden = n_hidden
        self.d = context_dim # number of data, dimension
        self.num_class = num_actions # number of actions
        self.M = M # number of particles
        
        
        self.num_vars = self.d * n_hidden + n_hidden * (1 + self.num_class) + self.num_class + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*num_class; b2 = num_class; 2 variances
        self.theta = np.zeros([self.M, self.num_vars])  # particles, will be initialized later
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.matrix('y') # labels
        mask = T.matrix('mask') # labels
        
        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        w_2 = T.matrix('w_2') # weights between hidden layer and output layer
        b_2 = T.vector('b_2') # bias of output
        
        N = T.scalar('N') # number of observations
        
        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')
        
        ###
        prediction = T.nnet.nnet.softmax(T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2)
        
        ''' define the log posterior distribution '''
        log_lik_data = T.sum(T.sum(y * mask * T.log(prediction))) + 1e-3 * log_gamma
        log_prior_w = -0.5 * (self.num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + (b_2**2).sum())  \
                       + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda
        
        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])
        
        # automatic gradient
        self.logp_gradient = theano.function(
             inputs = [X, y, mask, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda],
             allow_input_downcast=True
        )
        
        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction, allow_input_downcast=True)

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)

        print 'Successfully initialized the models!'

    '''
        Training with SVGD
    '''
    def train(self, X_train, y_train, weights, batch_size = 512, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        N0 = X_train.shape[0]  # number of observations
              
        grad_theta = np.zeros([self.M, self.num_vars])  # gradient 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        iter = 0
        batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
            dw1, db1, dw2, db2, dloggamma, dloglambda = self.logp_gradient(X_train[batch,:], y_train[batch], weights[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
            grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)
            
        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
        else:
            historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
        adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
        self.theta = self.theta + master_stepsize * adj_grad 
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, self.num_class)
        b2 = np.zeros((self.num_class,))
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)
    
    '''
        Pack all parameters in our model
    '''    
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2.flatten(), b2, [loggamma],[loglambda]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2 = np.reshape(w[:self.n_hidden*self.num_class],  [self.n_hidden, self.num_class])
        b2 = w[self.n_hidden*self.num_class:(self.n_hidden+1)*self.num_class]

        w = w[(self.n_hidden+1)*self.num_class:]
        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]
        
        return (w1, b1, w2, b2, loggamma, loglambda)

    def predict(self, X_test):
        pred_y_test = np.zeros([X_test.shape[0]])
        # '''
        #     Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        # '''
        # ########### Random sample from the posterior distributions ##############
        i = np.random.randint(self.M)

        w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
        pred_y_test = (self.nn_predict(X_test, w1, b1, w2, b2))
        pred = pred_y_test
        # pred_y_test = np.zeros([self.M, X_test.shape[0]])
        
        # for i in range(self.M):
        #     w1, b1, w2, b2, v11, v12, v21, v22, loggamma, loglambda, logphi, logpsi = self.unpack_weights(self.theta[i, :])
        #     pred_y_test[i, :] = (self.nn_predict(X_test, w1, b1, w2, b2, v11, v12, v21, v22) * self.std_y_train + self.mean_y_train).flatten()

        # pred = np.mean(pred_y_test, axis=0)

        return pred

class dgf_nn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, context_dim, num_actions, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1):
        self.n_hidden = n_hidden
        self.d = context_dim # number of data, dimension
        self.num_class = num_actions # number of actions
        self.M = M # number of particles
        
        
        self.num_vars = self.d * n_hidden + n_hidden * (1 + self.num_class) + self.num_class + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*num_class; b2 = num_class; 2 variances
        self.theta = np.zeros([self.M, self.num_vars])  # particles, will be initialized later
        self.oldtheta = np.zeros([self.M, self.num_vars])  # particles, will be initialized later
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.matrix('y') # labels
        mask = T.matrix('mask') # labels
        
        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        w_2 = T.matrix('w_2') # weights between hidden layer and output layer
        b_2 = T.vector('b_2') # bias of output
        
        N = T.scalar('N') # number of observations
        
        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')
        
        ###
        # prediction = T.nnet.nnet.softmax(T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2)
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2
        
        ''' define the log posterior distribution '''
        # log_lik_data = T.sum(T.sum(y * mask * T.log(prediction))) + 1e-3 * log_gamma

        log_lik_data = - T.sum(T.power(prediction - y, 2) * mask) + 1e-3 * log_gamma

        # log_prior_w = -0.5 * (self.num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + (b_2**2).sum())  \
        #                + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda
        
        priorprec = T.log(b0/a0)
        log_prior_w = -0.5 * (self.num_vars-2) * (T.log(2*np.pi)-priorprec) - (T.exp(priorprec)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + (b_2**2).sum())  \
                        + 1e-3* log_lambda

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])
        
        # automatic gradient
        self.logp_gradient = theano.function(
             inputs = [X, y, mask, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda],
             allow_input_downcast=True
        )
        
        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction, allow_input_downcast=True)

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)
            self.oldtheta[i, :] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)
        print 'Successfully initialized the models!'

    '''
        Training with SVGD
    '''
    def train(self, X_train, y_train, weights, batch_size = 512, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        N0 = X_train.shape[0]  # number of observations
        
        grad_theta = np.zeros([self.M, self.num_vars])  # gradient 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        iter = 0
        batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
            dw1, db1, dw2, db2, dloggamma, dloglambda = self.logp_gradient(X_train[batch,:], y_train[batch], weights[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
            grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)
            
        # calculating the kernel matrix
        if(self.M > 1):
            w2_kxy, w2_dxkxy = self.w2_kernel(h=-1)
            kxy, dxkxy = self.svgd_kernel(h=-1)
            grad_theta = (np.matmul(kxy, grad_theta) + 0.5 * w2_dxkxy) / self.M   # \Phi(x)
       ########### Stop Here ###############

        self.oldtheta = np.copy(self.theta)

        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
        else:
            historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
        adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
        self.theta = self.theta + master_stepsize * adj_grad 
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, self.num_class)
        b2 = np.zeros((self.num_class,))
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)
    
    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    def w2_kernel(self, h = -1):
        pairwise_dists = cdist(self.theta, self.oldtheta)**2
        # pdb.set_trace()
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp(-pairwise_dists / h**2 / 2) * (1-pairwise_dists / (h ** 2))

        dxkxy = -np.matmul(Kxy, self.oldtheta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i], sumkxy)
        dxkxy = dxkxy
        return (Kxy, dxkxy)
    
    '''
        Pack all parameters in our model
    '''    
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2.flatten(), b2, [loggamma],[loglambda]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2 = np.reshape(w[:self.n_hidden*self.num_class],  [self.n_hidden, self.num_class])
        b2 = w[self.n_hidden*self.num_class:(self.n_hidden+1)*self.num_class]

        w = w[(self.n_hidden+1)*self.num_class:]
        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]
        
        return (w1, b1, w2, b2, loggamma, loglambda)

    def predict(self, X_test):
        pred_y_test = np.zeros([X_test.shape[0]])
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        ########### Random sample from the posterior distributions ##############
        i = np.random.randint(self.M)
      
        w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
        
        pred_y_test = (self.nn_predict(X_test, w1, b1, w2, b2))
        pred = pred_y_test

        return pred
