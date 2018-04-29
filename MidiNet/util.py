import scipy.misc
import numpy as np
import pretty_midi

def save_images(images, size, image_path):
    return scipy.misc.imsave(image_path, merge(inverse_transform(images), size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def inverse_transform(images):
    return (images+1.)/2.

def save_midi(midi_array, image_path):
    pm_gen = pretty_midi.PrettyMIDI(initial_tempo=80)
    instrument = pretty_midi.Instrument(1, is_drum = False, name = "piano1")
    for t in range(len(midi_array)):
        v = np.max(midi_array[t])
        i = np.argmax(midi_array[t])
        note = pretty_midi.Note(velocity=(int)(v), pitch=i, start=t/10, end=(t + 1)/10)
        instrument.notes.append(note)
    pm_gen.instruments.append(instrument)
    pm_gen.write(image_path)