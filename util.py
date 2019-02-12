import numpy as np
import tensorflow as tf
import math

from constants import *
from midi_util import *

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def build_or_load(allow_load=True):
    from model import build_models
    models = build_models()
    models[0].summary()
    if allow_load:
        try:
            models[0].load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    modelIndex = 1;
    for model in models:
        print('Model ', modelIndex, ' : ', get_model_memory_usage(BATCH_SIZE, model), ' GB');
        modelIndex = modelIndex + 1;
    return models

def get_all_files(paths):
    potential_files = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(root, f)
                if os.path.isfile(fname) and fname.endswith('.mid'):
                    potential_files.append(fname)
    return potential_files

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.get_output_shape_at(-1):
            if s is None:
                continue
            if type(s) is tuple:
                for ss in s:
                    if ss is None:
                        continue
                    single_layer_mem *= ss;
            else:
                single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
