from pysndfx import AudioEffectsChain
import pandas as pd
import random
import librosa
import numpy as np
import sys
from datetime import datetime
import os
import re
import json

# UTILITY

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


# MAIN 

def randomize(d, e, exercise):
    
    store_new = {}
    for param, value in d.items():
        if param in effects_to_randomize.get(e):     
            upper, lower = value
            rand_val = random.randint(upper, lower)
            #  chance of 0 if classification
            store_new[param] = rand_val if exercise == 'regression' else random.choice([0, rand_val])
    return store_new
    
def simple_gen(soundfile, effects, exercise):
    """
    Simple augmentation
    
    """
    fx_manager = AudioEffectsChain()
    applied = {}
    for effect, parameters in effects.items():
        if exercise != 'regression':
            if effect not in effects_to_steady:
                
                # if want to turn all other features on / off 
                # check if its not required to be constant
                
                if int(random.choice([True, False])) == 0:
                    #  pass on effect completely
                    continue 
        fx = getattr(fx_manager, effect)
        if effect in effects_to_randomize:
            parameters = randomize(parameters, e = effect, exercise = exercise)
        fx(**parameters)
        applied[effect] = parameters
    augmented = fx_manager(soundfile)
    return (augmented, applied)

def normalize_arrays(inputs):
    
    # get largest shape
    inputs = np.atleast_2d(*inputs)
    max_shape = max([i.shape for i in inputs], key = lambda x: x[len(inputs[0] - 1)])

    def padder(inp, max_shape): 
        # Normalize shapes for concat
        zero_grid = np.zeros(max_shape)
        x,y = inp.shape
        zero_grid[:x, :y] = inp
        return zero_grid

    return [padder(ii, max_shape) for ii in inputs]

def make_tabular(samples):
    """
    Return tracks (NDarray) and features (DataFrame) in tabular form  
    """

    X, features = zip(*samples)
    X = normalize_arrays(X)
    stacked = np.stack(X)
    # Using dot notation for column names after unnesting records
    unnest = lambda x: flatten(x)
    effects_added = pd.DataFrame(list(map(unnest, features)))
    return (stacked, effects_added)

def sound_factory(soundfiles, batch_size, augment_n, effects, exercise, path = None):

    """
    Augment tracks in bulk, will transform later to keras type generator

    """

    while True:

        # take n samples from series object passed
        # soundfiles =  [path + x for x in soundfiles]
        batch_files = random.sample(soundfiles, k = batch_size)

        batch_inputs = []
        batch_outputs = []

        for soundfile in batch_files:    
            
            # Load once
            track = load_track(soundfile)

            augmented = [simple_gen(track, 
                                   effects,
                                    exercise
                                   ) for i in range(augment_n)]
        
            features, labels = make_tabular(augmented)

            if exercise == 'classification':
                # binarize each column
                labels = labels.where(labels == 0).fillna(1)

            batch_inputs.append(features)
            batch_outputs.append(labels)
        
        max_shape = max([x.shape for x in batch_inputs], key = lambda x: x[2])
        
        def padder(inp, max_shape): 
            # Normalize shapes for concat
            zero_grid = np.zeros(max_shape)
            x,y,z= inp.shape
            zero_grid[:x, :y, :z] = inp
            return zero_grid
        
        normalized_batch_inputs = [padder(x, max_shape) for x in batch_inputs]

        # not sure if I have to keep stacking
        batch_x = np.vstack(normalized_batch_inputs)
        # how do we represent outputs?
        batch_y = pd.concat(batch_outputs, axis = 0)

        yield (batch_x, batch_y)


# EFFECTS CONSTRAINT, only overdrive gain is randomized in this exercise - 
# we like the other effects to model typical user behaviour 

effects = {
    'overdrive' :  {'gain' : (10,80), 'colour' :  (0,20)}, # these are really the only two we randomize
    'reverb': {
            'reverberance' : (0,100),
            'hf_damping': 50,
            'pre_delay': 20,
            'reverberance': 50,
            'room_scale': 100,
            'stereo_depth': 100,
            'wet_gain': 0,
            'wet_only': False
                },
    'delay' :
            {'decays': [0.3, 0.25],
             'delays': [1000, 1800],
             'gain_in': 0.8,
             'gain_out': 0.5,
             'parallel': False}
}


# Which effects we want randomized if regression
effects_to_randomize = {'overdrive': ['gain', 'colour']}
# Which should remain constant if classification
effects_to_steady = {'overdrive': ['gain']} # 

pipeline_settings = {
    'effects' : effects,
    'exercise' : 'classification',
    'batch_size': 1, # number of audio tracks taken as part of a batch
    'augment_n' : 5 # number of augmentations per track
}

if __name__ == "__main__":

    _, *soundfiles = sys.argv

    # modify batch size to match files introduced
    pipeline_settings['batch_size'] = len(soundfiles)

    print (
        'Anticipated number of examples: {}'.format(
        pipeline_settings.get('batch_size') * pipeline_settings.get('augment_n')))

    print ('*' * 50)

    soundfiles_arg = soundfiles if isinstance(soundfiles, list) else [soundfiles]

    train_generator = sound_factory(soundfiles = soundfiles_arg, **pipeline_settings)

    train_X, train_Y = next(train_generator)

    dt = datetime.now().strftime('%Y-%m-%d')

    filenames = {
            'x' : dt + '_train_X_' + pipeline_settings.get('exercise') + '.npy',
            'y' : dt + '_train_Y_' + pipeline_settings.get('exercise') + '.csv'
    }

    print ('SAVING TO THE FOLLOWING FILES!\n')
    print (json.dumps(filenames, indent = 4))

    print ('Saving samples and labels!')

    np.save(filenames.get('x'), train_X)

    train_Y.to_csv(open(filenames.get('y'), 'w'))

