import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.callbacks import EarlyStopping, TensorBoard
import argparse
import midi
import os
from constants import *
from dataset import *
from generate import *
from midi_util import midi_encode
from model import *
import numpy as np
import constants
from metrics import metrics_eval, metrics_eval_single
import sys


def main():
    evaluate()

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluates music.')
    parser.add_argument('--batch', default=32, type=int, help='Number of batches to generate for evaluation')
    args = parser.parse_args()
    batch_size = args.batch
    print('Available styles : ', constants.styles)
    print('Loading data')
    train_data, train_labels = load_all(styles, BATCH_SIZE, SEQ_LEN, None)
    if (len(train_data) == 4):
      _, _, _, style_data = train_data
      train_data = None
      style_data = style_data[:,0,:]
      train_labels = train_labels[0][:,:,:,0]
      all_track = train_labels.reshape(train_labels.shape[0], train_labels.shape[1], train_labels.shape[2], 1)
      print('Dataset metrics : ')
      metrics_eval_single(all_track)
      print('-----------------------------------------------------------------------')
      all_tracks_style = {}
      for i in range(style_data.shape[1]):
        all_tracks_style[i] = []
      for i in range(len(style_data)):
        s = style_data[i]
        for k in range(len(s)):
          if s[k] == 1:
            all_tracks_style[k].append(all_track[i])
      flat_styles = []
      for i in range(len(constants.styles)):
        for j in range(len(constants.styles[i])):
          flat_styles.append(constants.styles[i][j])
      for i in range(len(all_tracks_style)):
        all_tracks_style[i] = np.array(all_tracks_style[i])
        print('Dataset Style "', flat_styles[i] ,'" metrics : ')
        metrics_eval_single(all_tracks_style[i])
        print('-----------------------------------------------------------------------')
      num_styles = len(all_tracks_style)
      all_tracks_style = None
      style_data = None
      all_track = None
      models = build_or_load()
      
      gen_track_style = {}
      gen_track = []
      for i in range(num_styles):
        gen_track_style[i] = []
      for j in range(batch_size):
          styles = [np.mean([one_hot(i, NUM_STYLES)], axis=0) for i in range(num_styles)]
          results = zip(*list(generate(models, SEQ_LEN // NOTES_PER_BAR, styles)))
          for k, result in enumerate(results):
            mf = unclamp_midi(result)
            bars = mf[:,MIN_NOTE:,0].reshape(mf.shape[0], NUM_NOTES)
            bars = bars.reshape(bars.shape[0], bars.shape[1], 1)
            gen_track_style[k].append(bars)
            gen_track.append(bars)
      gen_track = np.array(gen_track)
      print('Generation metrics : ')
      metrics_eval_single(gen_track)
      print('-----------------------------------------------------------------------')
      for i in range(len(gen_track_style)):
        gen_track_style[i] = np.array(gen_track_style[i])
        print('Generation Style "', flat_styles[i] ,'" metrics : ')
        metrics_eval_single(gen_track_style[i])
        print('-----------------------------------------------------------------------')
    else:
      _, _, _, style_data, condit_data = train_data
      train_data = None
      style_data = style_data[:,0,:]
      train_labels = train_labels[0][:,:,:,0]
      gen_condit_data = condit_data
      gen_cond_style = {}
      condit_data = condit_data[:,:,:,0]
      all_track = np.stack([condit_data, train_labels], axis = 3)
      print('Dataset metrics : ')
      metrics_eval(all_track)
      print('-----------------------------------------------------------------------')
      all_tracks_style = {}
      for i in range(style_data.shape[1]):
        all_tracks_style[i] = []
        gen_cond_style[i] = []
      for i in range(len(style_data)):
        s = style_data[i]
        for k in range(len(s)):
          if s[k] == 1:
            all_tracks_style[k].append(all_track[i])
            gen_cond_style[k].append(gen_condit_data[i])
      flat_styles = []
      for i in range(len(constants.styles)):
        for j in range(len(constants.styles[i])):
          flat_styles.append(constants.styles[i][j])
      for i in range(len(all_tracks_style)):
        all_tracks_style[i] = np.array(all_tracks_style[i])
        gen_cond_style[i] = np.array(gen_cond_style[i])
        print('Dataset Style "', flat_styles[i] ,'" metrics : ')
        metrics_eval(all_tracks_style[i])
        print('-----------------------------------------------------------------------')
      
      all_tracks_style = None
      all_track = None
      gen_condit_data = None
      train_labels = None
      style_data = None
      condit_data = None
      gen_seeds = {}
      for i in range(len(gen_cond_style)):
        choices = np.random.choice(gen_cond_style[i].shape[0], batch_size)
        choices = gen_cond_style[i][choices, :, :, :]
        gen_seeds[i] = choices
      gen_cond_style = None
      models = build_or_load()
      
      gen_track_style = {}
      gen_track = []
      for i in range(len(gen_seeds)):
        gen_track_style[i] = []
      for i in range(len(gen_seeds)):
        choices = gen_seeds[i]
        for j in range(len(choices)):
          styles = [np.mean([one_hot(i, NUM_STYLES)], axis=0)]
          seed = choices[j]
          seed = unclamp_midi(seed)
          results = zip(*list(generate_roll(models, SEQ_LEN // NOTES_PER_BAR, styles, seed)))
          for k, result in enumerate(results):
            mf = unclamp_midi(result)
            bars = mf[:,MIN_NOTE:,0].reshape(mf.shape[0], NUM_NOTES)
            results = np.stack([seed[:, MIN_NOTE:,0], bars], axis = 2)
            gen_track_style[i].append(results)
            gen_track.append(results)
      gen_track = np.array(gen_track)
      print('Generation metrics : ')
      metrics_eval(gen_track)
      print('-----------------------------------------------------------------------')
      for i in range(len(gen_track_style)):
        gen_track_style[i] = np.array(gen_track_style[i])
        print('Generation Style "', flat_styles[i] ,'" metrics : ')
        metrics_eval(gen_track_style[i])
        print('-----------------------------------------------------------------------')
      
      
if __name__ == '__main__':
    main()

