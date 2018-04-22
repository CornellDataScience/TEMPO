import tensorflow as tf
import numpy as np
from layer import *

class MidiNet(object):
    def __init__(self, sess, batch_size = 64):
        self.sess = sess
        self.batch_size = batch_size

        self.cond_bn0 = batch_norm(name = "cond_bn0")
        self.cond_bn1 = batch_norm(name = "cond_bn1")
        self.cond_bn2 = batch_norm(name = "cond_bn2")
        self.cond_bn3 = batch_norm(name = "cond_bn3")

        self.gen_bn0 = batch_norm(name = "gen_bn0")
        self.gen_bn1 = batch_norm(name = "gen_bn1")
        self.gen_bn2 = batch_norm(name = "gen_bn2")
        self.gen_bn3 = batch_norm(name = "gen_bn3")
        self.gen_bn4 = batch_norm(name = "gen_bn4")

    #Cond2d should be a [batchsize, 16, 128, 1] tensor
    def generator(self, noise, cond1d = None, cond2d = None):
        
        #Conditioner Architecture

        #cond_l0 of size [batchsize, 16, 1, 16]
        cond_l0 = lrelu(self.cond_bn0(conv(cond2d, 16, filter_h = 1, filter_w = 128, str_h = 1, str_w = 2, name = "cond_conv0")))
        #cond_l1 of size [batchsize, 8, 1, 16]
        cond_l1 = lrelu(self.cond_bn1(conv(cond_l0, 16, filter_h = 2, filter_w = 1, name = "cond_conv1")))
        #cond_l2 of size [batchsize, 4, 1, 16]
        cond_l2 = lrelu(self.cond_bn2(conv(cond_l1, 16, filter_h = 2, filter_w = 1, name = "cond_conv2")))
        #cond_l3 of size [batchsize, 2, 1, 16]
        cond_l3 = lrelu(self.cond_bn3(conv(cond_l2, 16, filter_h = 2, filter_w = 1, name = "cond_conv3")))

        #Integrate Conditions with everything else

        cond1dtensor = tf.reshape(cond1d, [self.batch_size, 1, 1, cond1d.get_shape().as_list()[-1]])
        noise = tf.concat([noise, cond1d], 1)

        l0 = tf.nn.relu(self.gen_bn0(linear(noise, 1024, scope = "gen_lin0")))
        l0 = tf.concat([l0, cond1d], 1)

        l1 = tf.nn.relu(self.gen_bn1(linear(l0, 256, scope = "gen_lin1")))
        l1 = tf.reshape(l1, [self.batch_size, 2, 1, 128])
        l1 = concat_1d(l1, cond1dtensor)
        l1 = concat_2d(l1, cond_l3)

        l2 = tf.nn.relu(self.gen_bn2(deconv(l1, [self.batch_size, 4, 1, 128], filter_h = 2, filter_w = 1, name = "gen_dconv2")))
        l2 = concat_1d(l2, cond1dtensor)
        l2 = concat_2d(l2, cond_l2)

        l3 = tf.nn.relu(self.gen_bn3(deconv(l2, [self.batch_size, 8, 1, 128], filter_h = 2, filter_w = 1, name = "gen_dconv3")))
        l3 = concat_1d(l3, cond1dtensor)
        l3 = concat_2d(l3, cond_l1)

        l4 = tf.nn.relu(self.gen_bn4(deconv(l3, [self.batch_size, 16, 1, 128], filter_h = 2, filter_w = 1, name = "gen_dconv4")))
        l4 = concat_1d(l4, cond1dtensor)
        l4 = concat_2d(l4, cond_l0)

        return tf.nn.sigmoid(deconv(l4, [self.batch_size, 16, 128, 1], filter_h = 1, filter_w = 128, str_h = 1, str_w = 2, name = "gen_dconv5"))