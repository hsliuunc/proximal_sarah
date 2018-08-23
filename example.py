'''
test the effectiveness of proximal sarah on mnist dataset
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from logistic_oracles import *
from prox_sarah import *


## load dataset
data = DataHolder(lam1=1e-4, lam2=0)
n = data.num_train_examples
p = data.data_dim

## initialize parameters
step_size = 0.1
num_epochs = 30
m = n
verbose = True
init_x = np.random.normal(0,1,p)


print '-----------------------------------------------'
print 'Training model...'
print '-----------------------------------------------\n'

f = open('../output/SarahOutputMNIST_stepsize_1','wb')
output_data = prox_sarah(data, step_size=1, max_epoch=num_epochs, m=m, x0=init_x, verbose=True)
cPickle.dump(output_data, f)
f.close()

f = open('../output/SarahOutputMNIST_stepsize_0.1','wb')
output_data = prox_sarah(data, step_size=0.1, max_epoch=num_epochs, m=m, x0=init_x, verbose=True)
cPickle.dump(output_data, f)
f.close()

f = open('../output/SarahOutputMNIST_stepsize_0.01','wb')
output_data = prox_sarah(data, step_size=0.01, max_epoch=num_epochs, m=m, x0=init_x, verbose=True)
cPickle.dump(output_data, f)
f.close()

print '-----------------------------------------------'
print 'Training complete'
print '-----------------------------------------------'



