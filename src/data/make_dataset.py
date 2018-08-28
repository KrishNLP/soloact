import sox
import random
import yaml
import os
import numpy as np
from inspect import getmembers, signature, isclass, isfunction, ismethod
import librosa
import librosa.display
import yaml
import tempfile
import glob
import logging
import pandas as pd
import itertools
import sys
random.seed(666)

def flatten(x, par = '', sep ='.'):
    """Flatten joining parent keys with dot"""
    store = {}
    for k,v in x.items():
        if isinstance(v,dict):
            store = {**store, **flatten(v, par =  par + sep + k if par else k)}
        else:
            store[par + sep + k] = v
    return store

def load_track(path, sr = 44100):

    """Unlikely we make modifications to this but nice to have it separately"""

    x, sr = librosa.load(path, sr = sr) # default
    return x

def rand(x,y):
    if all([v < 1 for v in [x,y]]):
        return random.uniform(x, y)
    else:
        return random.randint(x,y)

def validate_reduce_fx(effects):
    """Function for validating existence of effects to minimize error during augmentation"""
    FX = sox.Transformer()
    sox_arsenal = dict(getmembers(FX, predicate=lambda x: ismethod(x)))
    try:
        assert all(f in sox_arsenal for f in effects), 'Invalid methods provided'
        return True, effects
    except Exception as e:
        invalid = {f for f in effects if f not in sox_arsenal}
        return False, invalid

# def temp_to_array():
#     with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:

logger = logging.getLogger()
logger.setLevel('CRITICAL')

def feature_pipeline(arr, **kwargs):

    mfcc = librosa.feature.mfcc(arr,
            sr = 44100,
            n_mfcc = 26, **kwargs)
    mfcc_mean = np.mean(mfcc, axis = 0)
    return mfcc_mean


def pad(l_arrays):

    max_shape = max([x.shape for x in l_arrays], key = lambda x: x[0])

    def padder(inp, max_shape):
        # Normalize shapes for concat
        zero_grid = np.zeros(max_shape)
        x,y = inp.shape
        zero_grid[:x, :y] = inp
        return zero_grid

    reshapen = [padder(x, max_shape) for x in l_arrays]

    batch_x = np.array(reshapen)

    return batch_x


def augment_track(file, effects,
               exercise = 'regression',
               sustain = ['overdrive', 'reverb'],
               write = False
              ):
    """
    Rigid implementation of augmenation procedure

    """

    FX = sox.Transformer()

    labels = {}

    for effect, parameters in effects.items():

        if exercise == 'classification' and effect not in sustain:
            if int(random.choice([True, False])) == 0:
                print ('{} skipped!'.format(effect))
                continue # skip effect

        effect_f = getattr(FX, effect)
        f_defaults = signature(effect_f).parameters

        # effect defaults
        f_defaults = {k: f_defaults[k].default for k in f_defaults.keys()}

        used = {}

        for param, val in parameters.items():

            state = val.get('state')
            default = val.get('default') # boolean

            if state == 'constant':
                used[param] = f_defaults.get(param) if default is True else val.get('upper') # upper can be a list
            elif state == 'random':
                # retrieve bounds
                if not isinstance(f_defaults.get(param), list):
                    lower, upper = [val.get(bound) for bound in ['lower', 'upper']]
                    assert upper > lower, \
                       'Upper bound for {} must be greater than its lower bound'.format(effect + '.' + param)
                    used[param] = rand(lower, upper)
                    continue
                raise TypeError('Will not parse random list values!')

        effect_f(**used)
        labels[effect] = used

    if write is not False:
        FX.build(file, os.path.join(write, file.split('/')[-1]))
    else:
        with tempfile.NamedTemporaryFile(suffix = '.wav') as tmp:
            FX.build(file, tmp.name)
            array, sr = librosa.load(tmp.name, sr = 41000)
            FX.clear_effects()
            return flatten(labels), feature_pipeline(array)

def augment_data(write_path = False):


    DATA_DIR = '../../data/'

    config = 'config.yaml'
    config = yaml.load(open(config, 'r'))

    # SPLIT CONFIGURATIONS
    augmentation_config = config.get('DataAugmentation')
    pipeline_config = config.get('pipeline_config')

    # PREDETERMINED MODEL GUITAR SPLIT
    use_models_train = pipeline_config.get('train_models')
    use_models_test = pipeline_config.get('test_models')

    # ALL POWERCHORDS
    train_soundfiles = [glob.glob(DATA_DIR + 'interim/powerchords/' + mod + '/*.wav')
                    for mod in use_models_train]
    train_soundfiles = list(itertools.chain.from_iterable(train_soundfiles))

    # VALIDATE EFFECTS BEFORE STARTING AUGMENTATION CHAIN
    effects = augmentation_config.get('effects')
    valid, effects = validate_reduce_fx(effects)

    # REDUCE LIST TO ACTIVE ONLY
    effects = {k:v for k,v in effects.items() if k in augmentation_config.get('active')}

    # NOT A KEYWORD TO AUGMENTATION FUNCTION
    augmentation_config.pop('active')

    # REPLACE
    augmentation_config['effects'] = effects

    # DON'T TOUCH
    _n_augementations = 5

    if write_path is not False:

        print ('Are you sure you want to {} files to "{}"?'.format(
                len(train_soundfiles) * _n_augementations, write_path))

        print ('1 to proceed, any other key to terminate')
        if int(input()) == 1:
            # can't take relative path with join here
            augmentation_config['write'] = write_path
            # MAKE DIRECTORY
            os.makedirs(os.path.dirname(write_path + '/'), exist_ok=True)
        else:
            sys.exit('Operation cancelled')

    all_features = []
    all_labels = []

    for sf in train_soundfiles[:2]:

        store_all = []

        for i in range(_n_augementations):

            if write_path is not False:
                augment_track(sf, **augmentation_config) # call without store
            else:
                store_all.append(augment_track(sf, **augmentation_config))

        if write_path is False:

            labels, features = zip(*store_all)
            all_labels = all_labels + list(labels)
            all_features = all_features + list(features)

    all_features = [np.expand_dims(x, axis = 1) for x in all_features]
    all_features = pad(all_features)
    return all_features, pd.DataFrame(all_labels)

augment_data(write_path = None)
