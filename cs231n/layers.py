import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out=np.dot(x.reshape(len(x),-1),w)+b[None]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx=np.dot(dout,w.T)
  dx=dx.reshape(x.shape)
  dw=np.dot(x.reshape(len(x),-1).T,dout)
  # db=np.mean(dout,axis=0)
  db=np.sum(dout,axis=0) # because every xi in x dot product w plus b turn to be dout[i] ,so every b influence each dout[i],should sum it up
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out=np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  x[x>=0]=1
  x[x<0]=0
  dx=dout*x
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    x_oragin=x
    mean=np.mean(x,axis=0)
    var=np.var(x,axis=0)
    std=np.sqrt(var)
    running_mean=running_mean*momentum+(1-momentum)*mean
    running_var=running_var*momentum+(1-momentum)*std
    x=(x-mean)/std
    out=gamma*x+beta
    cache=(x,gamma,beta,mean,var+eps,x_oragin)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x=(x-running_mean)/running_var
    out=gamma*x+beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x,gamma,beta,mean,var,x_oragin=cache
  # dx=dout*gamma/std

  dgamma=np.sum(dout*x,axis=0)
  dbeta=np.sum(dout,axis=0)
  # dx=gamma*(1.0/np.sqrt(var))*dout-(x*dgamma+dbeta)*gamma*(1.0/np.sqrt(var))/x.shape[0]
  # dx=dout*gamma/std+np.sum(dout*gamma*(x_oragin-mean),axis=0)/(-0.5)*(1.0/std)**3*(x_oragin-mean)/x.shape[0]+np.sum(dout*gamma*(-1/std),axis=0)/x.shape[0]
  # dx=dout*gamma/np.sqrt(var)+np.sum(dout*gamma*(-0.5)*(1/np.sqrt(var)**3)*(x_oragin-mean),axis=0)*2*(x_oragin-mean)/x.shape[0]+np.sum(dout*gamma/np.sqrt(var)*(-1),axis=0)/x.shape[0]
  dx=(x.shape[0]*dout*gamma-np.sum(dout*gamma,axis=0)-x*np.sum(dout*gamma*x,axis=0))*(1./x.shape[0]*(1./np.sqrt(var)))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    tmp=np.random.uniform(0,1,x.shape)
    mask=(tmp>p)
    out=x*mask.astype(int)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out=x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx=dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  x_pad=np.pad(x,conv_param['pad'],'constant')[1:-1,1:-1,:,:]
  f,c,kh,kw=w.shape
  n,c,H,W=x_pad.shape
  if ((H-kh)/float(conv_param['stride'])-int((H-kh)/float(conv_param['stride'])))>0:
    print "can't get integer conv output"
  new_h,new_w=int((H-kh)/float(conv_param['stride']))+1,int((W-kw)/float(conv_param['stride']))+1
  w_matrix=w.reshape(-1,c*kh*kw)
  out=np.zeros([n,f,new_h,new_w])
  for i in range(new_h):
    for j in range(new_w):
      x_n=x_pad[:,:,i*conv_param['stride']:i*conv_param['stride']+kh,j*conv_param['stride']:j*conv_param['stride']+kw]
      x_n=x_n.reshape(-1,c*kh*kw).T
      tmp=np.dot(w_matrix,x_n)+b[:,None]
      out[:,:,i,j]=tmp.T

  # out=np.zeros([f,new_h,new_w])
  # for m in range(f):
  #   w_matrix=w[m].reshape[]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param=cache
  n,f,nh,nw=dout.shape
  f,c,kh,kw=w.shape
  # n,c,H,W=x.shape
  tmp=np.zeros(w.shape)
  x_pad=np.pad(x,conv_param['pad'],'constant')[1:-1,1:-1,:,:]
  dx=np.zeros(x_pad.shape)
  dw=np.zeros([c*kh*kw,f])
  db=np.zeros(f)
  for i in range(nh):
    for j in range(nw):
      dout_tmp=np.squeeze(dout[:,:,i,j])#n,f
      x_n=x_pad[:,:,i*conv_param['stride']:i*conv_param['stride']+kh,j*conv_param['stride']:j*conv_param['stride']+kw]#n,c
      x_n=x_n.reshape(-1,c*kh*kw).T#n
      w_matrix=w.reshape(-1,c*kh*kw).T#f
      dx_tmp=np.dot(w_matrix,dout_tmp.T).reshape(c,kh,kw,-1)
      dx_tmp=np.transpose(dx_tmp,(3,0,1,2))
      dx[:,:,i*conv_param['stride']:i*conv_param['stride']+kh,j*conv_param['stride']:j*conv_param['stride']+kw]+=dx_tmp
      dw+=np.dot(x_n,dout_tmp)
      db+=np.sum(dout_tmp,axis=0)
  dw=dw.T.reshape(-1,c,kh,kw)
  dx=dx[:,:,1:-1,1:-1]
  # dw=np.transpose(dw,(3,0,1,2))





  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:x_pad=np.pad(x,1,'constant')
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  x_pad=np.pad(x,1,'constant')[1:-1,1:1,:,:]
  kh,kw=pool_param['pool_height'],pool_param['pool_width']
  stride=pool_param['stride']
  N,C,H,W=x.shape
  if ((H-kh)/stride>int((H-kh)/stride)):
    print "stride and the kh is not match"
  nh,nw=(H-kh)/stride+1,(W-kw)/stride+1
  out=np.ones([N,C,nh,nw])
  for i in range(nh):
    for j in range(nw):
      tmp=x[:,:,i*stride:i*stride+stride,j*stride:j*stride+stride]
      x_col=tmp.reshape(N*C,-1)
      tmp=np.max(x_col,keepdims=True,axis=1)
      out[:,:,i,j]=tmp.reshape(N,C)



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """


  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param=cache
  dx=np.zeros(x.shape)
  N,C,nh,nw=dout.shape
  kh,kw=pool_param['pool_height'],pool_param['pool_width']
  stride=pool_param['stride']
  for i in range(nh):
    for j in range(nw):
      dout_col=dout[:,:,i,j]
      dout_col=dout_col.reshape(N*C,-1)
      x_tmp=x[:,:,i*stride:i*stride+kh,j*stride:j*stride+kw]
      x_tmp=x_tmp.reshape(N*C,-1)
      tmp=np.zeros([N*C,kw*kh])
      x_forward=np.max(x_tmp,axis=1,keepdims=True)
      dx_col=np.where(x_tmp==x_forward,dout_col,tmp)
      dx_part=np.reshape(dx_col,[N,C,kh,kw])
      dx[:,:,i*stride:i*stride+kh,j*stride:j*stride+kw]+=dx_part


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  if bn_param['mode']=='train':
    mean=np.mean(x,axis=(0,2,3),keepdims=True)
    var=np.var(x,axis=(0,2,3),keepdims=True)
    out=(x-mean)/(np.sqrt(var)+bn_param['eps'])
    out=out*gamma.reshape(mean.shape)+beta.reshape(mean.shape)
    bn_param['running_mean']=bn_param['running_mean']*bn_param['momentum']+mean*bn_param['momentum']
    bn_param['running_var']=bn_param['running_var']*bn_param['momentum']+var*bn_param['momentum']
  else:
    mean,var=bn_param['running_mean'],bn_param['running_var']
    out=(x-mean)/(np.sqrt(var)+bn_param['eps'])
    out=out*gamma.reshape(mean.shape)+beta.reshape(mean.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def  softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
