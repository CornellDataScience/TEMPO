import tensorflow as tf
from model import MidiNet
sess = tf.Session()

def main():
    mn = MidiNet(sess)
    print("all good")

if __name__ == "__main__": main()