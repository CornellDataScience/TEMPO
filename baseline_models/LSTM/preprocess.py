import os
import numpy as np
import pretty_midi as pmidi


TIME_STEPS = 40
N_SONGS = 100
PATH = 'MidiData/Nottingham/train/'


def midi_to_data(instrument):
    song = np.ndarray((len(instrument.notes), 4))
    last_note_start = 0
    for ix, n in enumerate(instrument.notes):
        # note : tone_length, pitch, intensity, time_since_last
        tone_length = n.end - n.start
        pitch = n.pitch / 127
        intensity = n.velocity / 127
        time_since_last = n.start - last_note_start
        last_note_start = n.start
        song[ix] = np.array([tone_length, pitch, intensity, time_since_last])


    time_steps = TIME_STEPS
    data_steps = len(song)-time_steps-1
    print(len(song))
    data = np.empty(data_steps, dtype=np.ndarray)
    labels = np.empty(data_steps, dtype=np.ndarray)
    for i in range(time_steps, len(song)):
        datum = song[i-time_steps: i]
        label = song[i]
        data[i-time_steps-1] = datum
        labels[i-time_steps-1] = label
    return data, labels

def get_instrument(pmidi_file):
    instr_list = pmidi_file.instruments
    instr_list.sort(key = lambda x: len(x.notes), reverse=True)
    for instr in instr_list:
        if instr.is_drum:
            continue
        else:
            return instr

data = np.empty(N_SONGS, dtype=object)
labels = np.empty(N_SONGS, dtype=object)
n = 0

for filename in os.listdir(PATH):
    if filename.endswith(".mid"):
        if n < N_SONGS:
            path = PATH + filename
            sample = pmidi.PrettyMIDI(path)
            instrument = get_instrument(sample)
            if len(instrument.notes) < 100:
                continue
            song_dat, song_labels = midi_to_data(instrument)
            data[n] = song_dat
            labels[n] = song_labels
            n += 1
            print(n)
        else:
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            break

np.save(open('train_data.npy', 'wb'), data)
np.save(open('train_labels.npy', 'wb'), labels)
