import tensorflow as tf
import numpy as np
from layer import *

class MidiNet(object):
    def __init__(self, sess, batch_size = 64, output_w = 16, output_h = 128, output_layers = 1, d1_dim = 13, d2_dim = 1):
        self.sess = sess
        self.batch_size = batch_size
        self.d1_dim = d1_dim
        self.d2_dim = d2_dim
        self.output_w = output_w
        self.output_h = output_h
        self.output_layers = output_layers

        self.cond_bn0 = batch_norm(name = "cond_bn0")
        self.cond_bn1 = batch_norm(name = "cond_bn1")
        self.cond_bn2 = batch_norm(name = "cond_bn2")
        self.cond_bn3 = batch_norm(name = "cond_bn3")

        self.gen_bn0 = batch_norm(name = "gen_bn0")
        self.gen_bn1 = batch_norm(name = "gen_bn1")
        self.gen_bn2 = batch_norm(name = "gen_bn2")
        self.gen_bn3 = batch_norm(name = "gen_bn3")
        self.gen_bn4 = batch_norm(name = "gen_bn4")

        self.dis_bn1 = batch_norm(name = "dis_bn1")
        self.dis_bn2 = batch_norm(name = "dis_bn2")
        self.dis_bn3 = batch_norm(name = "dis_bn3")

        self.build_model()

    def build_model(self):
        self.noise = tf.placeholder(tf.float32, [self.batch_size, 100], name = 'noise')
        self.cond1d = tf.placeholder(tf.float32, [self.batch_size, self.d1_dim], name = 'cond1d_ph')
        self.cond2d = tf.placeholder(tf.float32, [self.batch_size, self.output_w, self.output_h, self.output_layers], name = 'cond2d_ph')
        self.G = self.generator(self.noise, self.cond1d, self.cond2d)

        self.sample  = self.sampler(self.noise, self.cond1d, self.cond2d)
        
        self.data = tf.placeholder(tf.float32, [self.batch_size, self.output_w, self.output_h, self.output_layers], name = 'data_ph')
        self.D, self.D_logits, self.fm = self.discriminator(self.data, self.cond1d, self.cond2d, reuse = False)
        
        self.Df, self.Df_logits, self.fmf = self.discriminator(self.G, self.cond1d, self.cond2d, reuse = True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = 0.9 * tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Df_logits, labels = tf.zeros_like(self.Df)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Df_logits, labels = tf.ones_like(self.Df)))    


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

    #sample is of size [batch_size, 16, 128, 1]
    #cond2d is of size [batch_size, 16, 128, 1] 
    def discriminator(self, sample, cond1d, cond2d, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        cond1dtensor = tf.reshape(cond1d, [self.batch_size, 1, 1, cond1d.get_shape().as_list()[-1]])
        sample = concat_1d(sample, cond1dtensor)

        l0 = lrelu(conv(sample, 1 + self.d1_dim, filter_h = 1, filter_w = 128, name = "dis_conv0"))
        fm = l0
        l0 = concat_1d(l0, cond1dtensor)
        l0 = concat_2d(l0, cond2d)

        l1 = lrelu(self.dis_bn1(conv(l0, 64 + self.d1_dim, filter_h = 4, filter_w = 1, name = "dis_conv1")))
        l1 = tf.reshape(l1, [self.batch_size, -1])
        l1 = tf.concat([l1, cond1d], 1)

        l2 = lrelu(self.dis_bn2(linear(l1, 1024, scope = "dis_lin2")))
        l2 = tf.concat([l2, cond1d], 1)

        l3 = linear(l2, 1, scope = "dis_lin3")

        return tf.nn.sigmoid(l3), l3, fm

    def sampler(self, noise, cond1d = None, cond2d = None):
        tf.get_variable_scope().reuse_variables()
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
        