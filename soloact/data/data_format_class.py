import keras
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Input, Flatten, Dropout, Activation,  Dense, Embedding
from keras.callbacks import ModelCheckpoint
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import yaml
from functools import partial
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import re

# PATHS
base = ''
features_fp = os.path.join(base, 'training_X_power.npy')
labels_fp = os.path.join(base, 'sample_class.csv')
config_fp = os.path.join(base, 'config_power.yaml')

# DATA

predicted_effects = ['overdrive.gain_db', 'reverb.reverberance', 'chorus.delays']


def one_hot_binarize(labels, predict_effects):

    abbrv = [x.split('.') for x in predict_effects]
    abbrv = [(x[0][:2] + '.' + x[-1][:2]).upper() for x in abbrv]
    binarize = labels[predict_effects].fillna(0)
    binarize = binarize.where(binarize == 0).replace(np.nan, 1)
    def labelize(row, abbrv):
        return ':__'.join(e + str(int(r)) for e,r in zip(abbrv, list(tuple(row))))
    labels = binarize.apply(lambda x: labelize(x, abbrv), axis = 1)
    encoder = LabelEncoder()
    one_hot = np_utils.to_categorical(encoder.fit_transform(labels))
    states = list(encoder.classes_)
    return one_hot, states


def format_data(config=config_fp, predicted_effects=['overdrive.gain_db',
                                                         'reverb.reverberance',
                                                         'chorus.delays'] ):


    random_fx, active, sustain = find_on_effects(config_fp)
    print ('Active effects: {}'.format(active))
    print ('Persisting {} during augmentation'.format(sustain))
    print ('Parameters allowed randomization {}'.format(random_fx))

    features = globals()['features']
    labels = globals()['labels'][predicted_effects]

    omit = list_check(labels)

    [*data] = train_test_split(features, labels, shuffle=True, test_size=0.2, random_state=44)
    shapes = list(map(lambda x: x.shape, data))

    X_train, X_val, Y_train, Y_val = data

    # shared model, need binarized
    Y_train_bin, Y_train_states = one_hot_binarize(Y_train, predict_effects=predicted_effects)
    Y_val_bin, Y_val_states = one_hot_binarize(Y_val, predict_effects=predicted_effects)
    return (X_train, X_val, Y_train, Y_train_bin, Y_train_states, Y_val, Y_val_bin, Y_val_states, shapes)


class Dataset(object):

    def __init__(
            self,
            config,
            predicted_effects
            features,
            labels,
    ):
        self.config = config
        self.predicted_effects = predicted_effects
        self.features = features
        self.labels = labels

        self.make_dataset()

    def find_on_effects(self, path):
        effects = yaml.load(open(path, 'r'))
        used = effects.get('DataAugmentation')
        active = used.get('active')
        sustain = used.get('sustain')
        r = used.get('effects')
        random_subeffects = [[k for k,v in r.get(a).items() if v.get('state') == 'random'] for a in active]

        print ('Active effects: {}'.format(active))
        print ('Persisting {} during augmentation'.format(sustain))
        print ('Parameters allowed randomization {}'.format(random_fx))
        return {a: random_subeffects[ix] for ix, a in enumerate(active)}, active, sustain

    def list_check(self, labels, t = list):
        """Some parameters are given as lists"""
        labels_sub = labels.sample(n=10, random_state = 666).applymap(str)
        labels_sub = labels_sub.applymap(lambda x: True if re.search(r'\[', x) else False)
        labels_sub = labels_sub.any(axis = 0).loc[lambda x: x == True]
        return labels_sub.keys().tolist()

    def one_hot_binarize(labels):

        abbrv = [x.split('.') for x in self.predicted_effects]
        abbrv = [(x[0][:2] + '.' + x[-1][:2]).upper() for x in abbrv]
        binarize = labels[predict_effects].fillna(0)
        binarize = binarize.where(binarize == 0).replace(np.nan, 1)
        def labelize(row, abbrv):
            return ':__'.join(e + str(int(r)) for e,r in zip(abbrv, list(tuple(row))))
        labels = binarize.apply(lambda x: labelize(x, abbrv), axis = 1)
        encoder = LabelEncoder()
        one_hot = np_utils.to_categorical(encoder.fit_transform(labels))
        states = list(encoder.classes_)
        return one_hot, states

    def make_dataset(self):
        random_fx, active, sustain = self.find_on_effects(config_fp)

        omit = self.list_check(self.labels)

        data = train_test_split(features, labels, shuffle=True, test_size=0.2, random_state=44)

        data = {
            'X_train': data[0]
            'X_val': data[1]
            'y_train': data[2]
            'y_val': data[3]
        }

        for k, v in data.items():
            print('{} {}'.format(k, v.shape))

        data['y_train_bin'], data['y_train_states'] = self.one_hot_binarize(data['y_train'])
        data['y_val_bin'], data['y_val_states'] = self.one_hot_binarize(data['y_val'])

        #  scaling regression targets
        #  mean = 0, std = 1
        from sklearn.preprocessing import StandardScaler

        slr = StandardScaler()
        data['y_train'] = sclr.fit_transform(data['y_train'])
        data['y_val'] = sclr.transform(data['y_val'])

        self.data = data

        return self.data

features = np.load(features_fp)
labels = pd.read_csv(labels_fp, delimiter=",", index_col=0)

ds = Dataset(config, predicted_effects, features, labels)

data = ds.make_dataset()

def dump_pickle(obj, name):
    """
    Saves an object to a pickle file.

    args
        obj (object)
        name (str) path of the dumped file
    """
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    """
    Loads a an object from a pickle file.

    args
        name (str) path to file to be loaded

    returns
        obj (object)
    """
    with open(name, 'rb') as handle:
        return pickle.load(handle)

dump_pickle(ds, path)

ds = load_pickle(path)







