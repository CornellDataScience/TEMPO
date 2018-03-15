# adapted from https://arxiv.org/pdf/1511.06434.pdf

import tensorflow as tf
import numpy as np 


class DCGan(object): 
    def __init__ (self, a):
        self.hi = a

    def generator (self, z):
        mat = tf.Variable(tf.random_normal(shape = [100, 4 * 4 * 1024], stddev = 0.02))
        bias = tf.Variable(tf.constant(0.0, tf.float32, shape = [4 * 4 * 1024]))
        x = tf.matmul(z, mat) + bias
        x = tf.reshape(x, [1,4,4,1024])
        filter1 = tf.Variable(tf.random_normal(shape = [5,5,512,1024], stddev = 0.02))
        bias1 = tf.Variable(tf.constant(0.0, tf.float32, shape = [512]))
        x = tf.nn.conv2d_transpose(x, filter1, [1,8,8,512], [1,2,2,1], padding = 'SAME')
        x = tf.reshape(tf.nn.bias_add(x, bias1), [1,8,8,512])
        x = tf.nn.relu(x)
        filter2 = tf.Variable(tf.random_normal(shape = [5,5,256,512], stddev = 0.02))
        bias2 = tf.Variable(tf.constant(0.0, tf.float32, shape = [256]))
        x = tf.nn.conv2d_transpose(x, filter2, [1,16,16,256], [1,2,2,1], padding = 'SAME')
        x = tf.reshape(tf.nn.bias_add(x, bias2), [1,16,16,256])
        x = tf.nn.relu(x)
        filter3 = tf.Variable(tf.random_normal(shape = [5,5,128,256], stddev = 0.02))
        bias3 = tf.Variable(tf.constant(0.0, tf.float32, shape = [128]))
        x = tf.nn.conv2d_transpose(x, filter3, [1,32,32,128], [1,2,2,1], padding = 'SAME', name = None)
        x = tf.reshape(tf.nn.bias_add(x, bias3), [1,32,32,128])
        x = tf.nn.relu(x)
        filter4 = tf.Variable(tf.random_normal(shape = [5,5,3,128], stddev = 0.02))
        bias4 = tf.Variable(tf.constant(0.0, tf.float32, shape = [3]))
        x = tf.nn.conv2d_transpose(x, filter4, [1,64,64,3], [1,2,2,1], padding = 'SAME', name = None)
        x = tf.reshape(tf.nn.bias_add(x, bias4), [1,64,64,3])
        x = tf.nn.tanh(x)
        return x