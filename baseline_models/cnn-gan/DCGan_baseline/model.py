# adapted from https://arxiv.org/pdf/1511.06434.pdf

import tensorflow as tf
import numpy as np 


class DCGan(object): 
    """
    Args:
    batch is the batch size
    """
    def __init__ (self, batch_size = 64):
        self.batch_size = batch_size

    """
    Returns a generative model that generates a batch of images from noise

    Args: 
    z is a 2-D tensor of shape [self.batch_size, 100]
    """
    def generator (self, z):
        with tf.variable_scope("generator"):
            mat = tf.Variable(tf.random_normal(shape = [100, 4 * 4 * 1024], stddev = 0.02))
            bias = tf.Variable(tf.constant(0.0, tf.float32, shape = [4 * 4 * 1024]))
            x = tf.matmul(z, mat) + bias
            x = tf.reshape(x, [self.batch_size,4,4,1024])
            filter1 = tf.Variable(tf.random_normal(shape = [5,5,512,1024], stddev = 0.02))
            bias1 = tf.Variable(tf.constant(0.0, tf.float32, shape = [512]))
            x = tf.nn.conv2d_transpose(x, filter1, [self.batch_size,8,8,512], [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias1), [self.batch_size,8,8,512])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = tf.nn.relu(x)
            filter2 = tf.Variable(tf.random_normal(shape = [5,5,256,512], stddev = 0.02))
            bias2 = tf.Variable(tf.constant(0.0, tf.float32, shape = [256]))
            x = tf.nn.conv2d_transpose(x, filter2, [self.batch_size,16,16,256], [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias2), [self.batch_size,16,16,256])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = tf.nn.relu(x)
            filter3 = tf.Variable(tf.random_normal(shape = [5,5,128,256], stddev = 0.02))
            bias3 = tf.Variable(tf.constant(0.0, tf.float32, shape = [128]))
            x = tf.nn.conv2d_transpose(x, filter3, [self.batch_size,32,32,128], [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias3), [self.batch_size,32,32,128])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = tf.nn.relu(x)
            filter4 = tf.Variable(tf.random_normal(shape = [5,5,3,128], stddev = 0.02))
            bias4 = tf.Variable(tf.constant(0.0, tf.float32, shape = [3]))
            x = tf.nn.conv2d_transpose(x, filter4, [self.batch_size,64,64,3], [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias4), [self.batch_size,64,64,3])
            x = tf.nn.tanh(x)
            return x
    """
    Returns a model which determines whether or not an image is real

    Args:
    images is an 4-D tensor of size [self.batch_size, 64, 64, 3] that are the images. 
    """
    def discriminator(self, image):
        with tf.variable_scope("discriminator"):
            filter1 = tf.Variable(tf.random_normal(shape = [5,5,3, 64], stddev = 0.02))
            bias1 = tf.Variable(tf.constant(0.0, tf.float32, shape = [64]))
            x = tf.nn.conv2d(image, filter1, strides = [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias1), [self.batch_size,32,32,64])
            x = leakyrelu(x)
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            filter2 = tf.Variable(tf.random_normal(shape = [5,5,64,128], stddev = 0.02))
            bias2 = tf.Variable(tf.constant(0.0, tf.float32, shape = [128]))
            x = tf.nn.conv2d(x, filter2, strides = [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias2), [self.batch_size, 16, 16, 128])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = leakyrelu(x)
            filter3 = tf.Variable(tf.random_normal(shape = [5, 5, 128, 256], stddev = 0.02))
            bias3 = tf.Variable(tf.constant(0.0, tf.float32, shape = [256]))
            x = tf.nn.conv2d(x, filter3, strides = [1,2,2,1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias3), [self.batch_size, 8, 8, 256])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = leakyrelu(x)
            filter4 = tf.Variable(tf.random_normal(shape = [5, 5, 256, 512], stddev = 0.02))
            bias4 = tf.Variable(tf.constant(0.0, tf.float32, shape = [512]))
            x = tf.nn.conv2d(x, filter4, strides = [1, 2, 2, 1], padding = 'SAME')
            x = tf.reshape(tf.nn.bias_add(x, bias4), [self.batch_size, 4, 4, 512])
            x = tf.layers.batch_normalization(x, momentum = 0.9, epsilon = 1e-5)
            x = leakyrelu(x)
            w1 = tf.Variable(tf.random_normal(shape = [4 * 4 * 512, 1], stddev = 0.02))
            b1 = tf.Variable(tf.constant(0.0, tf.float32, shape = [1]))
            x = tf.matmul(tf.reshape(x, [self.batch_size, 4 * 4 * 512]), w1) + b1
            return tf.nn.sigmoid(x), x

"""
Leaky relu activation with factor 0.2
"""
def leakyrelu(x, factor = 0.2):
        return tf.maximum(factor * x, x)