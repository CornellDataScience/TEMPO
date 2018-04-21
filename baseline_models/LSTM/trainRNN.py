import time
import random
import numpy as np
import tensorflow as tf
import pretty_midi as pmidi

"""Trains an LSTM model to be able to predict the next note based on the previous
(param : num_steps) notes. See the method lstm_drop for the final network.
The networks found in the rnn and lstm methods do not perform well.

The current configuration trains a 3 layer LSTM with 32 hidden units for each layer.
The network employs dropout with a keep probability of .5 followed by a fully
connected layer that connects to the 4 node outputself.

The data format resembles raw midi. It is an array of size 4 that is formatted
as [tone_length, pitch, intensity, time_since_last] where pitch and intensity
are normalized to be between 0 and 1 and tone_length and time_since_last
are in seconds.

Model is saved with the name given by the network_name parameter. Model is
to be loaded in generate.py to generate new music.
"""

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

n_neurons = 32
n_layers = 3
n_outputs = 4
n_hidden1 = 16
n_hidden2 = 64
num_steps = 40
learning_rate = .0001


train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

seed = train_data[0]

data_arr = (train_data, train_labels)

def rnn():
    X = tf.placeholder(tf.float32, [None, num_steps, n_outputs])
    y = tf.placeholder(tf.float32, [None,  n_outputs])

    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
          for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    states_concat = tf.concat(axis=1, values=states)
    hidden1 = tf.layers.dense(states_concat, n_hidden1, name="hidden1",
                               activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                               activation=tf.nn.relu)
    logits = tf.squeeze(tf.layers.dense(hidden2, n_outputs, name='outputs'))
    preds = tf.nn.relu(logits)

    error = tf.losses.mean_squared_error(labels=tf.squeeze(y), predictions=logits)
    loss = tf.reduce_mean(error)
    return X, y, preds, loss, logits

def lstm():
    X = tf.placeholder(tf.float32, [None, num_steps, n_outputs])
    y = tf.placeholder(tf.float32, [None,  n_outputs])

    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
          for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    top_layer_h_state = states[-1][1]
    hidden1 = tf.layers.dense(top_layer_h_state, n_hidden1, name="hidden1",
                               activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                               activation=tf.nn.relu)
    logits = tf.squeeze(tf.layers.dense(hidden2, n_outputs, name='outputs'))
    preds = tf.nn.relu(logits)

    error = tf.losses.mean_squared_error(labels=tf.squeeze(y), predictions=logits)
    loss = tf.reduce_mean(error)
    return X, y, preds, loss, logits

def lstm_drop():
    X = tf.placeholder(tf.float32, [None, num_steps, n_outputs])
    y = tf.placeholder(tf.float32, [None,  n_outputs])
    keep_prob = tf.placeholder(tf.float32,())

    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
          for layer in range(n_layers)]
    cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
              for cell in layers]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)


    top_layer_h_state = states[-1][1]
    #hidden1 = tf.layers.dense(top_layer_h_state, n_hidden1, name="hidden1",
    #                           activation=tf.nn.relu)
    #hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
    #                           activation=tf.nn.relu)
    logits = tf.squeeze(tf.layers.dense(top_layer_h_state, n_outputs, name='outputs'))
    preds = tf.nn.relu(logits)

    error = tf.losses.mean_squared_error(labels=tf.squeeze(y), predictions=logits)
    loss = tf.reduce_mean(error)
    return X, y, preds, loss, logits, keep_prob


n_epochs = 2
batch_size = 25
network_name = 'rnn_model_3x32_lstm'

def run():
    X, y, preds, loss, logits, keep_prob = lstm_drop()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        saver.save(sess, 'SavedModels/' + network_name)
        start_train = time.time()
        for epoch in range(n_epochs):
            for j in range(len(data_arr[0]) // batch_size):
                x_batch, y_batch = next_batch(batch_size, data_arr[0], data_arr[1])
                loss_val = sess.run([loss, training_op], feed_dict={X: x_batch, y: y_batch, keep_prob:.5})
            print('Epoch:', epoch,'  ', "{0:.2f}".format(((time.time() - start_train)/60)), 'mins')
            print('Loss:', loss_val)
            #print(states)
            #print(y.eval())
            #print(preds.eval())
        #save_path = saver.save(sess, "SavedModels/model.ckpt")
        #print("Model saved in path: %s" % save_path)



if __name__ == '__main__':
    run()
