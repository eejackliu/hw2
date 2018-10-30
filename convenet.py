import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from cs231n.layer_utils import *
class Convnet(object):
    # (conv - relu - 2x2 max pool )*3- affine - relu - affine - softmax
    def __init__(self,hidden_filters,input_dim=(3,32,32),filter_size=7,hidden_dim=100,num_class=10,
                 weight_scale=1e-3,reg=0.,dtype=np.float32):
        self.params={}
        self.reg=reg
        self.dtype=dtype
        C,H,W=input_dim
        f1,f2,f3=hidden_filters
        self.params['W1']=np.random.randn(f1,C,H,W)*weight_scale
        self.params['b1']=np.zeros(f1)
        self.params['W2']=np.random.randn(f2,f1,H/2,W/2)*weight_scale
        self.params['b2']=np.zeros(f2)
        self.params['W3']=np.random.randn(f3,f2,H/2**2,W/2**2)*weight_scale
        self.params['b3']=np.zeros(f3)
        self.params['W4']=np.random.randn(f3*(H/2**3)*2,hidden_dim)*weight_scale
        self.params['b4']=np.zeros(hidden_dim)
        self.params['W5']=np.random.randn(hidden_dim,num_class)*weight_scale
        self.params['b5']=np.zeros(num_class)
    def loss(self,X,y):
        w1,b1=self.params['W1'],self.params['b1']
        w2,b2=self.params['W2'],self.params['b2']
        w3,b3=self.params['W3'],self.params['b3']
        w4,b4=self.params['W4'],self.params['b4']
        w5,b5=self.params['W5'],self.params['b5']
        filter_size=

