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
    def __init__(self,hidden_filters,input_dim=(3,32,32),filter_size=3,hidden_dim=100,num_class=10,
                 weight_scale=1e-3,reg=0.,dtype=np.float32):
        self.params={}
        self.reg=reg
        self.dtype=dtype
        C,H,W=input_dim
        f1,f2,f3=hidden_filters
        self.params['W1']=np.random.randn(f1,C,filter_size,filter_size)*weight_scale
        self.params['b1']=np.zeros(f1)
        self.params['gamma1']=np.ones(f1)
        self.params['beta1']=np.zeros(f1)
        self.params['W2']=np.random.randn(f2,f1,filter_size,filter_size)*weight_scale
        self.params['b2']=np.zeros(f2)
        self.params['gamma2']=np.ones(f2)
        self.params['beta2']=np.zeros(f2)
        self.params['W3']=np.random.randn(f3,f2,filter_size,filter_size)*weight_scale
        self.params['b3']=np.zeros(f3)
        self.params['gamma3']=np.ones(f3)
        self.params['beta3']=np.zeros(f3)
        self.params['W4']=np.random.randn(f3*(H/2**3)**2,hidden_dim)*weight_scale
        self.params['b4']=np.zeros(hidden_dim)
        self.params['W5']=np.random.randn(hidden_dim,num_class)*weight_scale
        self.params['b5']=np.zeros(num_class)
        # self.params_bn={}
        # self.params_bn["bn_param1"]=self.params_bn["bn_param2"]=self.params_bn["bn_param3"]=
        self.bn_param= [{'mode': 'train', 'eps': 1e-5, 'momentum': 0.9, 'running_mean': 0,
                                           'running_var': 0}]*3
    def conv_batch(self,x,w,b,conv_param,pool_param=None,gamma=None,beta=None,bn_param=None):
        x,cache=conv_forward_fast(x,w,b,conv_param)
        bn,cache_bn=spatial_batchnorm_forward(x,gamma,beta,bn_param)
        re,cache_re=relu_forward(bn)
        p,cache_p=max_pool_forward_fast(re,pool_param)
        return p,(cache,cache_bn,cache_re,cache_p)
    def conv_batch_back(self,dout,cache):
        cache,cache_bn,cache_re,cache_p=cache
        dp=max_pool_backward_fast(dout,cache_p)
        dre=relu_backward(dp,cache_re)
        dbn,dgamma,dbeta=spatial_batchnorm_backward(dre,cache_bn)
        dx,dw,db=conv_backward_fast(dbn,cache)
        return dx,dw,db,dgamma,dbeta
    def loss(self,X,y=None):
        w1,b1=self.params['W1'],self.params['b1']
        w2,b2=self.params['W2'],self.params['b2']
        w3,b3=self.params['W3'],self.params['b3']
        w4,b4=self.params['W4'],self.params['b4']
        w5,b5=self.params['W5'],self.params['b5']
        gamma1,beta1=self.params['gamma1'],self.params['beta1']
        gamma2,beta2=self.params['gamma2'],self.params['beta2']
        gamma3,beta3=self.params['gamma3'],self.params['beta3']
        # bn_param1=self.params_bn["bn_param1"]
        # bn_param2=self.params_bn["bn_param2"]
        # bn_param3=self.params_bn["bn_param3"]



        filter_size=3
        conv_param={'stride':1,'pad':(filter_size-1)/2}
        pool_param={'pool_height':2,'pool_width':2,'stride':2}
        if y is None:
            for i in self.bn_param:
                i['mode']='test'
        else:
            for i in self.bn_param:
                i['mode']='train'
            # bn_param1['mode']=bn_param2['mode']=bn_param3['mode']='test'
            # self.params_bn["bn_param1"]['mode']=self.params_bn["bn_param2"]['mode']=self.params_bn["bn_param3"]['mode']='test'


        # x1,cache_x1=conv_relu_pool_forward(X,w1,b1,conv_param,pool_param)
        # out1,cache_bn1=spatial_batchnorm_forward(x1,gamma1,beta1,bn_param1)
        # x2,cache_x2=conv_relu_pool_forward(out1,w2,b2,conv_param,pool_param)
        # out2,cache_bn2=spatial_batchnorm_forward(x2,gamma2,beta2,bn_param2)
        # x3,cache_x3=conv_relu_pool_forward(out2,w3,b3,conv_param,pool_param)
        # out3,cache_bn3=spatial_batchnorm_forward(x3,gamma3,beta3,bn_param3)

        x1,cache_x1=self.conv_batch(X,w1,b1,conv_param,pool_param,gamma1,beta1,self.bn_param[0])
        x2,cache_x2=self.conv_batch(x1,w2,b2,conv_param,pool_param,gamma2,beta2,self.bn_param[1])
        x3,cache_x3=self.conv_batch(x2,w3,b3,conv_param,pool_param,gamma3,beta3,self.bn_param[2])
        x4,cache_x4=affine_relu_forward(x3,w4,b4)
        x5,cache_x5=affine_forward(x4,w5,b5)


        if y is None:
            return x5





        loss,dx=softmax_loss(x5,y)
        loss+=0.5*self.reg*(np.sum(w1**2)+np.sum(w2**2)+np.sum(w3**2)+np.sum(w4**2)+np.sum(w5**2))
        grad={}
        dx5,dw5,db5=affine_backward(dx,cache_x5)
        dx4,dw4,db4=affine_relu_backward(dx5,cache_x4)
        dx3,dw3,db3,dgamm3,dbeta3=self.conv_batch_back(dx4,cache_x3)
        dx2,dw2,db2,dgamm2,dbeta2=self.conv_batch_back(dx3,cache_x2)
        dx1,dw1,db1,dgamm1,dbeta1=self.conv_batch_back(dx2,cache_x1)
        grad['W5'],grad['b5']=dw5+self.reg*w5,db5
        grad['W4'],grad['b4']=dw4+self.reg*w4,db4
        grad['W3'],grad['b3'],grad['gamma3'],grad['beta3']=dw3+self.reg*w3,db3,dgamm3,dbeta3
        grad['W2'],grad['b2'],grad['gamma2'],grad['beta2']=dw2+self.reg*w2,db2,dgamm2,dbeta2
        grad['W1'],grad['b1'],grad['gamma1'],grad['beta1']=dw1+self.reg*w1,db1,dgamm1,dbeta1
        return loss,grad


data=get_CIFAR10_data()
small_data=get_CIFAR10_data(num_training=1000,num_validation=500)

model=Convnet((32,32,32),weight_scale=1e-4)
solver=Solver(model,data=small_data,num_epochs=20,batch_size=50,update_rule='adam',optim_config={"learning":1e-4},verbose=True,print_every=50)
solver.train()
# need to use torch to test why three layer net work is bad than the one layer