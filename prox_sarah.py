'''
main function: proximal sarah
'''

from logistic_oracles import *
import random, time
import numpy as np

#!/usr/bin/env python

def prox_sarah(data, step_size, max_epoch=50, m=0, x0=None, verbose=False):
    """
    :param data: dataset
    :param steo_size:
    :param max_epoch:
    :param m: inner loop iteration, set m=data size by default
    :param x0: initial point
    :param verbose:
    :return: optimal point, epochs, wall_times, train_error, test_error
    """

    n = data.num_train_examples
    d = data.data_dim
    lam1 = data.lam1
    lam2 = data.lam2

    start_time = time.time()

    epochs = []
    wall_times = []
    train_error = []
    test_error = []

    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    ## loop starts here:
    print('Proximal Sarah starts here: ')
    for k in xrange(max_epoch):
        epochs.append(k+1)
        v = data.logistic_batch_grad(range(n), x)
        x_old = x.copy()
        x = soft_thresh(x_old-step_size*v, step_size*lam1)
        func_value = data.logistic_batch_func(range(n), x_old)

        ## output in each epoch
        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Func. value: %.6e' % \
                     (k, step_size, func_value)
            print(output)

        for j in xrange(m):
            idx = random.randrange(n)
            grad_old_idx = data.logistic_batch_grad([idx],x_old)
            grad_cur_idx = data.logistic_batch_grad([idx],x)
            v = grad_cur_idx - grad_old_idx + v
            x_old = x
            x = soft_thresh(x_old-step_size*v,step_size*lam1)

        wall_times +=[time.time() - start_time]
        train_error += [func_value]

    output_data = {'epochs': epochs, 'wall_times': wall_times, 'train_error': train_error}

    return output_data


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)