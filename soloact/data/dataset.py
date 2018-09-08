import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import yaml
from functools import partial
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from keras.utils import np_utils

class Dataset(object):

    def __init__(
            self,
            config,
            predicted_effects,
            features,
            labels, scale = False):
        self.config = config
        self.predicted_effects = predicted_effects
        if not self.predicted_effects:
            raise ValueError('Please select with effects and sub effects you want to predict')
        print (self.predicted_effects)
        self.features = features
        self.labels = labels
        self.scaled = scale

    def find_on_effects(self, path):

        """
        Stdout effects used in pipeline

        """

        effects = yaml.load(open(path, 'r'))
        used = effects.get('DataAugmentation')
        active = used.get('active')
        sustain = used.get('sustain')
        r = used.get('effects')
        random_subeffects = [[k for k,v in r.get(a).items() if v.get('state') == 'random'] for a in active]

        print ('Active effects: {}'.format(active))
        print ('Persisting {} during augmentation'.format(sustain))
        print ('Parameters allowed randomization {}'.format(random_subeffects))
        return {a: random_subeffects[ix] for ix, a in enumerate(active)}, active, sustain

    def list_check(self, labels, t = list):

        """
        Return list of columns where cell type is list
        """

        sample_size = int(labels.shape[0] * 0.01) # ~ 1 % sample
        labels_sub = labels.sample(n=sample_size, random_state = 666).applymap(str)
        labels_sub = labels_sub.applymap(lambda x: True if re.search(r'\[', x) else False)
        labels_sub = labels_sub.any(axis = 0).loc[lambda x: x == True]
        self.list_labels = labels_sub.keys().tolist()
        return self.list_labels

    def one_hot_binarize(self, labels):

        """
        Returns one-hot on/off state combinations as dataframe for classification model

        args
            labels (dataframe)
        """

        abbrv = [x.split('.') for x in self.predicted_effects]
        abbrv = [(x[0][:2] + '.' + x[-1][:2]).upper() for x in abbrv]
        binarize = labels[self.predicted_effects].fillna(0)
        binarize = binarize.where(binarize == 0).replace(np.nan, 1)
        def labelize(row, abbrv):
            return ':__'.join(e + str(int(r)) for e,r in zip(abbrv, list(tuple(row))))
        labels = binarize.apply(lambda x: labelize(x, abbrv), axis = 1)
        encoder = LabelEncoder()
        one_hot = np_utils.to_categorical(encoder.fit_transform(labels))
        states = list(encoder.classes_)
        return one_hot, states

    def make_dataset(self):

        """
        Scale and reduce data for classification and regression tasks

        """

        if isinstance(self.labels, str) and os.path.isfile(self.labels):
            self.labels = pd.read_csv(self.labels, index_col = 0)

        if isinstance(self.features, str) and os.path.isfile(self.features):
            self.features = np.load(self.features)

        random_fx, active, sustain = self.find_on_effects(self.config)

        # return columns with list type values
        self.array_type_cols = self.list_check(self.labels)

        data = train_test_split(self.features, self.labels[self.predicted_effects], shuffle=True, test_size=0.2, random_state=44)

        data = {
            'X_train': data[0],
            'X_val': data[1],
            'y_train': data[2],
            'y_val': data[3],
        }

        print ('----- OUTPUT SHAPES -----')
        print ('\n'.join(("{}, {}".format(k, v.shape) for k,v in data.items())))

        data['y_train_bin'], data['y_train_states'] = self.one_hot_binarize(data['y_train'])
        data['y_val_bin'], data['y_val_states'] = self.one_hot_binarize(data['y_val'])

        # regression targets should exclude cells with list types
        data['y_train'] = data['y_train'].drop(self.array_type_cols, axis=1)
        data['y_val'] = data['y_val'].drop(self.array_type_cols, axis=1 )
        #  scaling regression targets
        #  mean = 0, std = 1
        if self.scaled:
            data['y_train'].replace(np.nan, 0, inplace = True)
            data['y_val'].replace(np.nan, 0, inplace = True)

            sclr = StandardScaler()
            data['y_train'] = sclr.fit_transform(data['y_train'])
            data['y_val'] = sclr.transform(data['y_val'])

        self.data = data

        return self.data
