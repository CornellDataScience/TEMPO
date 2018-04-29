import tensorflow as tf
import numpy as np
from layer import *
from scipy.ndimage.interpolation import shift
from sklearn.utils import shuffle
from util import save_images, save_midi
import time

class MidiNet(object):
    def __init__(self, sess, batch_size = 64, output_w = 16, output_h = 128, output_layers = 1, d1_dim = 13, d2_dim = 1, 
            lambda1 = 0.01, lambda2 = 0.1, sample_size = 72):
        self.sess = sess
        self.batch_size = batch_size
        self.d1_dim = d1_dim
        self.d2_dim = d2_dim
        self.output_w = output_w
        self.output_h = output_h
        self.output_layers = output_layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sample_size = sample_size

        self.cond_bn0 = batch_norm(name = "gen_cond_bn0")
        self.cond_bn1 = batch_norm(name = "gen_cond_bn1")
        self.cond_bn2 = batch_norm(name = "gen_cond_bn2")
        self.cond_bn3 = batch_norm(name = "gen_cond_bn3")

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
        self.noise_sum = tf.summary.histogram("noise", self.noise)
        self.cond1d = tf.placeholder(tf.float32, [self.batch_size, self.d1_dim], name = 'cond1d_ph')
        self.cond2d = tf.placeholder(tf.float32, [self.batch_size, self.output_w, self.output_h, self.output_layers], name = 'cond2d_ph')
        self.G = self.generator(self.noise, self.cond1d, self.cond2d)

        self.sample  = self.sampler(self.noise, self.cond1d, self.cond2d)
        
        self.data = tf.placeholder(tf.float32, [self.batch_size, self.output_w, self.output_h, self.output_layers], name = 'data_ph')
        self.D, self.D_logits, self.fm = self.discriminator(self.data, self.cond1d, reuse = False)
        
        self.Df, self.Df_logits, self.fmf = self.discriminator(self.G, self.cond1d, reuse = True)

        self.D_sum = tf.summary.histogram("D", self.D)
        self.Df_sum = tf.summary.histogram("D false", self.Df)
        self.G_sum = tf.summary.histogram("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = 0.9 * tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Df_logits, labels = tf.zeros_like(self.Df)))
        self.g_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Df_logits, labels = tf.ones_like(self.Df)))    

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.mean_g = tf.reduce_mean(self.G, axis = 0)
        self.mean_data = tf.reduce_mean(self.data, axis = 0)
        self.mean_loss = tf.multiply(tf.nn.l2_loss(self.mean_g - self.mean_data), self.lambda1)

        self.g_features = tf.reduce_mean(self.fmf, axis = 0)
        self.data_features = tf.reduce_mean(self.fm, axis = 0)
        self.feature_loss = tf.multiply(tf.nn.l2_loss(self.g_features - self.data_features), self.lambda2)

        self.d_loss = self.d_loss_fake + self.d_loss_real
        self.g_loss = self.g_loss0 + self.mean_loss + self.feature_loss

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        trainable_vars = tf.trainable_variables()

        self.d_vars = [var for var in trainable_vars if 'dis_' in var.name]
        self.g_vars = [var for var in trainable_vars if 'gen_' in var.name]

        self.checkbool = set(self.d_vars).intersection(self.g_vars) == set(trainable_vars)

    def train(self, config):
        X = np.load(config.training_data)
        Prev = np.load(config.prev_data)
        ycond = np.zeros(shape = [self.batch_size, 13])

        X, Prev = shuffle(X, Prev, random_state=0)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        self.sess.run(tf.global_variables_initializer())

        self.g_sum = tf.summary.merge([self.noise_sum, self.Df_sum, 
        self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.noise_sum, self.D_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0
        start_time = time.time()

        for epoch in range(config.epochs):
            batch_idxs = len(X) // self.batch_size
            for idx in range(batch_idxs):
                batch_images = X[idx*self.batch_size:(idx+1)*self.batch_size]
                prev_batch_images = Prev[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = ycond.tolist()
                batch_noise = sample_Z([self.batch_size, 100])

                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={self.data: batch_images, self.noise: batch_noise ,self.cond1d: batch_labels, self.cond2d:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.data: batch_images, self.noise: batch_noise ,self.cond1d :batch_labels, self.cond2d:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # We've tried to run more d_optim and g_optim, while getting a better result by running g_optim twice in this MidiNet version.
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.data: batch_images, self.noise: batch_noise, self.cond1d: batch_labels, self.cond2d:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.sess.run(self.d_loss_fake, feed_dict = {self.noise: batch_noise, self.cond1d:batch_labels, self.cond2d: prev_batch_images})
                errD_real = self.sess.run(self.d_loss_real, feed_dict = {self.data: batch_images, self.cond1d:batch_labels })
                errG = self.sess.run(self.g_loss, feed_dict = {self.data: batch_images, self.noise: batch_noise, self.cond1d: batch_labels, self.cond2d: prev_batch_images})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))
                    
            samples, d_loss, g_loss = self.sess.run(
                [self.sample, self.d_loss, self.g_loss],
                feed_dict={self.noise: batch_noise, self.data: batch_images, self.cond1d : batch_labels, self.cond2d: prev_batch_images }
            )
            #samples = (samples+1.)/2.
            save_images(samples[:5,:], [1, 5],
                        '{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            sound_sample = samples[0, :, :, 0]
            save_midi(sound_sample, '{}/train_{:02d}_{:04d}.mid'.format(config.sample_dir, epoch, idx))

            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
            % (epoch + 1, 
                time.time() - start_time, (errD_fake+errD_real)))

    #Cond2d should be a [batchsize, 16, 128, 1] tensor
    def generator(self, noise, cond1d = None, cond2d = None):
        with tf.variable_scope("generator"):
            #Conditioner Architecture

            #cond_l0 of size [batchsize, 16, 1, 16]
            cond_l0 = lrelu(self.cond_bn0(conv(cond2d, 16, filter_h = 1, filter_w = 128, str_h = 1, str_w = 2, name = "gen_cond_conv0")))
            #cond_l1 of size [batchsize, 8, 1, 16]
            cond_l1 = lrelu(self.cond_bn1(conv(cond_l0, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv1")))
            #cond_l2 of size [batchsize, 4, 1, 16]
            cond_l2 = lrelu(self.cond_bn2(conv(cond_l1, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv2")))
            #cond_l3 of size [batchsize, 2, 1, 16]
            cond_l3 = lrelu(self.cond_bn3(conv(cond_l2, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv3")))

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
    def discriminator(self, sample, cond1d, reuse = False):
        with tf.variable_scope("discriminator") as scope: 
            if reuse:
                scope.reuse_variables()
            cond1dtensor = tf.reshape(cond1d, [self.batch_size, 1, 1, cond1d.get_shape().as_list()[-1]])
            sample = concat_1d(sample, cond1dtensor)

            l0 = lrelu(conv(sample, 1 + self.d1_dim, filter_h = 1, filter_w = 128, name = "dis_conv0"))
            fm = l0
            l0 = concat_1d(l0, cond1dtensor)

            l1 = lrelu(self.dis_bn1(conv(l0, 64 + self.d1_dim, filter_h = 4, filter_w = 1, name = "dis_conv1")))
            l1 = tf.reshape(l1, [self.batch_size, -1])
            l1 = tf.concat([l1, cond1d], 1)

            l2 = lrelu(self.dis_bn2(linear(l1, 1024, scope = "dis_lin2")))
            l2 = tf.concat([l2, cond1d], 1)

            l3 = linear(l2, 1, scope = "dis_lin3")

            return tf.nn.sigmoid(l3), l3, fm

    def sampler(self, noise, cond1d = None, cond2d = None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            #Conditioner Architecture

            #cond_l0 of size [batchsize, 16, 1, 16]
            cond_l0 = lrelu(self.cond_bn0(conv(cond2d, 16, filter_h = 1, filter_w = 128, str_h = 1, str_w = 2, name = "gen_cond_conv0")))
            #cond_l1 of size [batchsize, 8, 1, 16]
            cond_l1 = lrelu(self.cond_bn1(conv(cond_l0, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv1")))
            #cond_l2 of size [batchsize, 4, 1, 16]
            cond_l2 = lrelu(self.cond_bn2(conv(cond_l1, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv2")))
            #cond_l3 of size [batchsize, 2, 1, 16]
            cond_l3 = lrelu(self.cond_bn3(conv(cond_l2, 16, filter_h = 2, filter_w = 1, name = "gen_cond_conv3")))

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
        