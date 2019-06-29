import numpy as np
import tensorflow as tf
from collections import deque
import midi
import argparse
import pretty_midi
import constants
from constants import *
from util import *
from dataset import *
from tqdm import tqdm
from midi_util import midi_encode, empty_timesteps_s, midi_multi_encode, midi_roll_read
from midi_write import save_midis, write_piano_roll_to_midi, load_midi_roll


class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, style, melody_roll=None, default_temp=1):
        self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        self.melody_roll = melody_roll
        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        if self.melody_roll is not None:
            if (self.melody_roll.shape[0] < SEQ_LEN):
                diff_len = SEQ_LEN - self.melody_roll.shape[0]
                lim = self.melody_roll.shape[0]
            else:
                diff_len = 0
                lim = SEQ_LEN
            self.cond_note = self.melody_roll[0:lim, MIN_NOTE:MAX_NOTE, :]
            self.cond_note = np.pad(self.cond_note, ((0, diff_len), (0, 0), (0, 0)), mode='constant',constant_values=0)
        else:
            self.cond_note = np.zeros((SEQ_LEN, NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR

        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp

    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )

    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:]),
            np.array(self.cond_note)
        )

    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)

        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # Apply volume
            self.next_note[n, 2] = vol
            # Flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1

    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp

        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        if self.melody_roll is not None:
            if self.melody_roll.shape[0] > t + 1:
                if (self.melody_roll.shape[0] < t + 1 + SEQ_LEN):
                    diff_len = t + 1 + SEQ_LEN - self.melody_roll.shape[0]
                    lim = self.melody_roll.shape[0]
                else:
                    diff_len = 0
                    lim = SEQ_LEN
                self.cond_note = self.melody_roll[t + 1:t + 1 + lim, MIN_NOTE:MAX_NOTE, :]
                self.cond_note = np.pad(self.cond_note, ((0, diff_len), (0, 0), (0, 0)), mode='constant',constant_values=0)
            else:
                self.cond_note = np.zeros((SEQ_LEN, NUM_NOTES, NOTE_UNITS))
        else:
            self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]

def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / (prob - 1))
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, styles, melody_file):
    print('Generating with styles:', styles)
    if not MELODY_GENERATION:
        melody_roll = midi_roll_read(melody_file)
    else:
        melody_roll = None
    _, time_model, note_model = models
    generations = [MusicGeneration(style, melody_roll) for style in styles]

    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def write_file(name, results):
    """
    Takes a list of all notes generated per track and writes it to file
    """
    results = zip(*list(results))
    nums = 0
    for i, result in enumerate(results):
        fpath = os.path.join(SAMPLES_DIR, name + '_' + str(i) + '.mid')
        print('Writing file', fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        mf = unclamp_midi(result)
        bars = mf[:,:,0].reshape(1, mf.shape[0], mf.shape[1])
        #mf = midi_encode(unclamp_midi(result))
        #midi.write_midifile(fpath, mf)
        diff_length = 128 - MAX_NOTE
        bars = np.concatenate((bars, np.zeros((bars.shape[0], bars.shape[1], diff_length))), axis=2)
        write_piano_roll_to_midi(bars, fpath)
        nums = i + 1
    return nums

  
def main():
    parser = argparse.ArgumentParser(description='Generates music.')
    parser.add_argument('--bars', default=32, type=int, help='Number of bars to generate')
    parser.add_argument('--styles', default=None, type=int, nargs='+', help='Styles to mix together')
    parser.add_argument('--output_file', default='output', type=str, help='output file name')
    parser.add_argument('--melody_file', default='melody.mid', type=str, help='Melody file to generate chords for if chord generation model')
    parser.add_argument('--combined_file', default='combined', type=str, help='Combined file of Melody and Chords in case of chord generation')
    args = parser.parse_args()

    models = build_or_load()
    print('Available styles : ', constants.styles)
    styles = [compute_genre(i) for i in range(len(genre))]

    if args.styles:
        # Custom style
        styles = [np.mean([one_hot(i, NUM_STYLES) for i in args.styles], axis=0)]

    nums = write_file(args.output_file, generate(models, args.bars, styles, args.melody_file))
    if not MELODY_GENERATION:
        melody_roll = load_midi_roll(args.melody_file)
        for i in range(nums):
          chords_roll = load_midi_roll(os.path.join(SAMPLES_DIR, args.output_file + '_' + str(i) + '.mid'))
          print(melody_roll.shape)
          print(chords_roll.shape)
          if chords_roll.shape[0] > melody_roll.shape[0]:
            diff_len = chords_roll.shape[0] - melody_roll.shape[0]
            melody_roll = np.pad(melody_roll, ((0, diff_len), (0, 0)), mode='constant',constant_values=0)
          else:
            diff_len = melody_roll.shape[0] - chords_roll.shape[0]
            chords_roll = np.pad(chords_roll, ((0, diff_len), (0, 0)), mode='constant',constant_values=0)
          tracks_roll = np.stack([melody_roll, chords_roll], axis=2)
          empty_timesteps_s(tracks_roll)
          fpath = os.path.join(SAMPLES_DIR, args.combined_file + '_' + str(i) + '.mid')
          print('Writing file', fpath)
          save_midis(tracks_roll, fpath)
if __name__ == '__main__':
    main()