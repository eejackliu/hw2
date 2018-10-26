import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape
#%%
x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = conv_forward_naive(x, w, b, conv_param)
correct_out = np.array([[[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]]])

# Compare your output to ours; difference should be around 1e-8
print 'Testing conv_forward_naive'
print 'difference: ', rel_error(out, correct_out)
# #%%
# from scipy.misc import imread, imresize
#
# kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# # kitten is wide, and puppy is already square
# d = kitten.shape[1] - kitten.shape[0]
# kitten_cropped = kitten[:, d/2:-d/2, :]
#
# img_size = 200   # Make this smaller if it runs too slow
# x = np.zeros((2, 3, img_size, img_size))
# x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
# x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
#
# # Set up a convolutional weights holding 2 filters, each 3x3
# w = np.zeros((2, 3, 3, 3))
#
# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
# w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
# w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
#
# # Second filter detects horizontal edges in the blue channel.
# w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#
# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = np.array([0, 128])
#
# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
#
# def imshow_noax(img, normalize=True):
#     """ Tiny helper to show images as uint8 and remove axis labels """
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0 * (img - img_min) / (img_max - img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
#
# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_noax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_noax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_noax(out[0, 1])
# plt.title('Edges')
# plt.subplot(2, 3, 4)
# imshow_noax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_noax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_noax(out[1, 1])
# plt.show()
#%%
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
print 'Testing conv_backward_naive function'
print 'dx error: ', rel_error(dx, dx_num)
print 'dw error: ', rel_error(dw, dw_num)
print 'db error: ', rel_error(db, db_num)