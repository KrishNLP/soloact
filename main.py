from setuptools import find_packages, setup
import os
import yaml


import xml.etree.ElementTree as ET
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import fnmatch
import sox
import glob

ROOT_DIR = os.getcwd() + '/'
DATA_RAW = 'data/raw/IDMT-SMT-GUITAR_V2/dataset1'
DATA_DIR = os.path.join(ROOT_DIR, DATA_RAW) + '/'
ANNOTATION_PATH = '*/annotation/*.xml'
AUDIO_PATH = '*/audio/*.wav'

def write_annotation():

    """
    Writing meta annotations

    https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html

    """
    # Operates at a subdirectory level

    import re
    subdirectories = glob.glob(DATA_DIR + ANNOTATION_PATH)
    filt = lambda fn: re.search(r'Major|Minor', fn) is None
    files = list(filter(filt, subdirectories))
    all_meta = []
    for filepath in files:
        guitar_model = re.sub(DATA_DIR, '', filepath).split('/')[0]
        tree = ET.parse(filepath)
        root = tree.getroot()
        record = {}
        record['guitarModel'] = guitar_model
        record['filepath'] = filepath
        # COMMON FLOW FOR ITERATING OVER XML
        for meta_attribute in root:
            # two fields
            if meta_attribute.tag == 'globalParameter':
                for field in meta_attribute:
                    record[field.tag] = field.text
            else:
                # FILE DATA
                for field in meta_attribute.find('event'):
                    record[field.tag] = field.text
        track_path = filepath.replace('annotation', 'audio')
        track_path = '/'.join(track_path.split('/')[:-1]) + '/'
        record['audioFileName'] = track_path + record['audioFileName']
        all_meta.append(record)
    to_df = pd.DataFrame(all_meta)
    filename = ROOT_DIR + 'data/interim/' + 'file_meta.csv'
    to_df.to_csv(filename)
    print ('Completed write to "{}"'.format(filename))
    return ''

import sox

def make_powerchords():

    annotations = ROOT_DIR + 'data/interim/file_meta.csv'
    powerchords_path = ROOT_DIR + 'data/interim/powerchords'
    try:
        annotations_df = pd.read_csv(annotations)
    except:
        write_annotation()
        annotations_df =  pd.read_csv(annotations)

    records = annotations_df.set_index(['guitarModel', 'pitch'])['audioFileName'].to_dict()
    sub_directories = []
    powerchords = {}

    for ii in annotations_df[['guitarModel', 'pitch', 'audioFileName']].itertuples():

        pitch = ii.pitch
        model = ii.guitarModel
        sd = powerchords_path + '/' + model

        if sd not in sub_directories:
            sub_directories.append(sd + '/')

        base = ii.audioFileName

        fifth = records.get((model, pitch + 7))
        octave = records.get((model, pitch + 12))

        if any([x is None for x in [fifth, octave]]):
            continue

        else:
            power_structure = '_'.join([str(int(pitch) + x) for x in [0,7,12]])
            filename =  sd +  '/pwch_' +  power_structure + '.wav'
            powerchords[filename] = [base, fifth, octave]

    for sd in sub_directories:
        os.makedirs(os.path.dirname(sd), exist_ok=True)

    for filename in powerchords:
        cbn = sox.Combiner()
        cbn.build(powerchords.get(filename), filename, 'mix')

make_powerchords()
