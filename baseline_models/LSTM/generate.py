import time
import random
import numpy as np
import tensorflow as tf
import pretty_midi as pmidi
from trainRNN import lstm_drop


def output_to_midi(out_seq, tempo=80):
    pm = pmidi.PrettyMIDI(initial_tempo=tempo)
    inst = pmidi.Instrument(program=42, is_drum=False)
    pm.instruments.append(inst)
    last_note_start = 0
    for note in out_seq:
        # note : tone_length, pitch, intensity, time_since_last
        start = last_note_start + note[3]
        end = start + note[0]
        pitch =  int(round(max(0, min(note[1]*127, 127))))
        velocity = int(round(note[2]*127))
        inst.notes.append(pmidi.Note(velocity, pitch, start, end))
        last_note_start = start
    return pm


train_data = np.load('train_data.npy')

seed = train_data[0]

n_outputs = 4
num_cheat_outputs = 50
output_len = 1000

def generate(model):
    X, y, preds, loss, logits, keep_prob = lstm_drop()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saved_model = tf.train.import_meta_graph(model + '.meta')
        saved_model.restore(sess, tf.train.latest_checkpoint('SavedModels/'))
        output = np.zeros((output_len, n_outputs))
        n = 0
        prev_notes = seed
        while (n < output_len):
            logs, res = sess.run([logits, preds], feed_dict={X: [prev_notes], keep_prob:1.})
            print(logs)
            output[n] = np.squeeze(res)
            if n < num_cheat_outputs:
                prev_notes = train_data[n]
            else:
                prev_notes = np.vstack([prev_notes[1:], res])
            n += 1
        mid = output_to_midi(output)
        mid.write('out.mid')

if __name__ == '__main__':
    generate('SavedModels/rnn_model_3x32_lstm')
