import tensorflow as tf

def lrelu(x, factor = 0.2):
    return tf.maximum(x, 0.2 * x)

def linear(inp, out_size, scope, stddev = 0.02, bias_start = 0.0):
    shape = inp.get_shape().as_list()
    with tf.variable_scope(scope):
        matrix = tf.get_variable("Weights", shape = [shape[1], out_size], 
            initializer= tf.random_normal_initializer(stddev = stddev))
        bias = tf.get_variable("Bias", shape = [out_size], 
            initializer = tf.constant_initializer(bias_start))
        return tf.matmul(inp, matrix) + bias

def conv(inp, out_size, filter_h = 5, filter_w = 5, str_h = 2, str_w = 2, stddev = 0.02, name = "conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("Weights", shape = [filter_h, filter_w, inp.get_shape()[-1].value, out_size], 
            initializer = tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inp, w, strides = [1, str_h, str_w, 1], padding = "VALID")
        bias = tf.get_variable("Bias", shape = [out_size], initializer= tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)
        return conv

def deconv(inp, out_shape, filter_h = 5, filter_w = 5, str_h = 2, str_w = 2, stddev = 0.02, name = "deconv2d", padding = "VALID"):
    with tf.variable_scope(name):
        #Note that deconv2d have [h, w, output, input] instead of normal conv2d
        w = tf.get_variable("Weights", shape = [filter_h, filter_w, out_shape[-1], inp.get_shape()[-1].value],
            initializer = tf.random_normal_initializer(stddev = stddev))
        deconv = tf.nn.conv2d_transpose(inp, w, output_shape = out_shape, strides = [1, str_h, str_w, 1], padding = padding)
        bias = tf.get_variable("Bias", shape = [out_shape[-1]], initializer= tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, bias)
        return deconv


# Concatenates the 1d condition vector onto the tensor. 
# Assumes that cond1d has shape [batch_size, 1, 1, _ ]
def concat_1d(tensor, cond1d):
    tensorshape = tensor.get_shape().as_list()
    condshape = cond1d.get_shape().as_list()
    return tf.concat([tensor, cond1d * tf.ones(shape = [tensorshape[0], tensorshape[1], tensorshape[2], condshape[3]])], axis = 3)

# Concatenates the 2d conditions with the tensor
def concat_2d(tensor, cond2d):
    tensorshape = tensor.get_shape().as_list()
    condshape = cond2d.get_shape().as_list()
    return tf.concat([tensor, cond2d * tf.ones(shape = [tensorshape[0], tensorshape[1], tensorshape[2], condshape[3]])], axis = 3)

# Batchnormalization
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
        
    def __call__(self, x):
        return tf.layers.batch_normalization(x, momentum= self.momentum, epsilon= self.epsilon, scale= True, name= self.name)
    