import os
import yaml
import xml.etree.ElementTree as ET
import pandas as pd
import json
import fnmatch
import glob
import argparse
import sox

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

strategies = {
    'power': ('powerchords', 'power_strat.yaml'),
    'septa' : ('septachords', 'septa_strat.yaml'),
    'triad' : ('triad', 'triad_strat.yaml')
}

def write_chords(strategy = 'power', write = False):

    # indexed_strategies = list(enumerate(list(strategies.keys())))

    assert strategy in strategies, '{} is not a valid strategy'.format(strategy)

    bp = os.path.abspath(os.path.join(ROOT_DIR,'data/interim/'))
    annotations = bp  + '/file_meta.csv'
    annotations_df = pd.read_csv(annotations, index_col = 0)

    chord_dir, strat_fn = strategies.get(strategy)
    strat_config = yaml.load(open(strat_fn, 'r'))
    strategy_path = os.path.abspath(os.path.join(bp, chord_dir))

    records = annotations_df.set_index(['guitarModel', 'pitch'])['audioFileName'].to_dict()

    sub_directories = []

    file_ticker = 0
    for ii in annotations_df[['guitarModel', 'pitch', 'audioFileName']].itertuples():

        pitch = ii.pitch
        model = ii.guitarModel
        audioname = ii.audioFileName.split('/')[-1].strip('.wav')

        subd =  os.path.join(strategy_path, model)
    #
        if subd not in sub_directories:
            os.makedirs(subd + '/', exist_ok = True)

        for chord, segment in strat_config.items():
            for s, pitch_components in segment.items():
                [*bindings], [*components] = zip(*[(records.get((model, pitch + x)), str(pitch + x)) for x in pitch_components])
                if any([fn is None for fn in bindings]) is False:
                    rename = '_'.join([audioname, chord, s] + components) + '.wav'
                    rename = os.path.join(subd, rename)
                    if write is True:
                        combiner = sox.Combiner()
                        combiner.build(bindings, rename, 'mix')
                        file_ticker += 1
                    else:
                        pass
                else:
                    break

STRATEGIES_AVAIALBLE = ['power', 'septa', 'triad']

write_chords(write = False, strategy = 'power')
