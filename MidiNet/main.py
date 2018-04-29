import tensorflow as tf
import os
from model import MidiNet
sess = tf.Session()

flags = tf.app.flags
flags.epochs = 20
flags.beta1 = 0.5
flags.learning_rate = 0.00005
flags.training_data = "beethoven_data.npy"
flags.prev_data = "beethoven_prev.npy"
flags.sample_dir = "sample"
def main():
    mn = MidiNet(sess)
    mn.train(flags)

if __name__ == "__main__": main()