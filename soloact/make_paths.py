from os.path import join, abspath
from glob import glob
from functools import partial
from collections import namedtuple, OrderedDict
from fnmatch import fnmatch

def make_source_paths(base):

    # PATHS
    data_path = partial(join,  *[base, 'data'])
    strat_path = partial(join, *[base, 'soloact', 'data', 'strategies'])
    config_path = partial(join, *[base, 'soloact', 'data', 'configurations'])
    models_path = partial(join,  *[base, 'models'])
    reports_path = partial(join, *[base, 'reports'])

    # MAJOR FILE TYPE CLASS
    data_directory = namedtuple('directory', 'trace extension meta type')

    raw = data_directory(*[data_path('raw/IDMT-SMT-GUITAR_V2/dataset1/'), '*/audio/*.wav', 'raw', '' ])
    interim = data_directory(*[data_path('interim/'), '', 'interim', ''])
    processed = data_directory(*[data_path('processed/'), '', 'processed', ''])

    configurations = glob(config_path() + '/*.yaml')
    # use configuration as benchmark, working order
    primary_config = config_path() + '/config.yaml'
    strategies = glob(strat_path() + '/*.yaml')
    pitch_names = [s.split('/')[-1].replace('_strat.yaml', '') + 's' for s in strategies]
    strategies = dict(list(zip(pitch_names, strategies)))
    annotations = data_directory(*[raw.trace[:-1] + '/*/annotation/', '*.xml', '', 'annotations'])

    chord_paths = [data_directory(*[data_path('interim', f), '/*/*.wav', f, ''])
                for f in pitch_names] + [raw]

    augmentation_paths = [data_directory(*[data_path('interim', f, 'augmented'), '', f, ''])
                for f in pitch_names] + [data_directory(*[data_path(raw.trace, 'augmented'), '', 'raw', ''])]

    guitar_models = ['Fender Strat Clean Neck SC',
                    'Ibanez Power Strat Clean Bridge HU',
                    'Ibanez Power Strat Clean Bridge+Neck SC', 'Ibanez Power Strat Clean Neck HU']

    store_inner = locals()
    store_inner.pop('data_directory')
    return store_inner
