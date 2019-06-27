"""
Preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing
import os
from constants import *
from midi_util import load_midi, empty_timesteps
from util import *

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])

def compute_genre(genre_id):
    """ Computes a vector that represents a particular genre """
    genre_hot = np.zeros((NUM_STYLES,))
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    styles_in_genre = len(styles[genre_id])
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot

def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def load_all(styles, batch_size, time_steps, mydrive=None):
    """
    Loads all MIDI files as a piano roll.
    (For Keras)
    """
    note_data = []
    beat_data = []
    style_data = []
    chosen_data = []
    note_target = []


    genre_map = {8: 0, 20: 1, 24: 2, 27: 3, 29: 4, 33: 5}
    train_label = np.load(LABELS_PATH)
    labels = np.zeros((train_label.shape[0], 6))
    for i in range(len(train_label)):
      cnt = np.count_nonzero(train_label[i] == 1)
      if cnt > 1 or cnt == 0:
        labels[i][5] = 1
        continue;
      for j in range(len(train_label[i])):
        if train_label[i][j] == 1:
          if j in genre_map:
            labels[i][genre_map[j]] = 1
          else:
            labels[i][5] = 1
    train_label = labels
    train = np.load(PIANOROLL_PATH)
    empty_timesteps(train)
    for i in range(len(train)):
        style_hot = train_label[i,:]
        if MELODY_GENERATION:
            seq = train[i,:,:,0]
            seq = adapt_pianroll(seq)
        else:
            seq = train[i,:int(train[i].shape[0] * 0.95),:,1]
            seq = adapt_pianroll(seq)
            seq_c = train[i,:int(train[i].shape[0] * 0.95),:,0]
            seq_c = adapt_pianroll(seq_c)
        if seq is None:
                continue
        if len(seq) >= time_steps:
            # Clamp MIDI to note range
            seq = clamp_midi(seq)
            # Create training data and labels
            if MELODY_GENERATION:
                train_data, label_data = stagger(seq, time_steps)
                note_data += train_data
                note_target += label_data
                chosen_data += label_data
            else:
                seq_c = clamp_midi(seq_c)
                train_data, label_data = stagger(seq, time_steps)
                train_data_c, label_data_c = stagger(seq_c, time_steps)
                note_data += train_data
                note_target += label_data
                chosen_data += label_data_c
            beats = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
            beat_data += stagger(beats, time_steps)[0]

            style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    note_data = np.array(note_data)
    beat_data = np.array(beat_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)
    chosen_data = np.array(chosen_data)
    return [note_data, chosen_data, beat_data, style_data], [note_target]

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')