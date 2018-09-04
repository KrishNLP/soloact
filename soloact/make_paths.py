from os.path import join, abspath

def make_source_paths(base):
    # DATA_DIR = abspath('../../data/')

    DATA_DIR = join(base, 'data')

    SOURCES = {
        'DATA_DIR': DATA_DIR,
        'power':{'trace' : join(DATA_DIR + '/interim/powerchords/'),
                 'ext': ''},
        'septa':{'trace' : join(DATA_DIR + '/interim/septachords/'),
                 'ext': ''},
        'triad':{'trace' : join(DATA_DIR + '/interim/triad/'),
                 'ext': ''},
        'sn':{'trace' : join(DATA_DIR + '/raw/IDMT-SMT-GUITAR_V2/dataset1/'),
                 'ext': 'audio/'},
        'config': join(base, 'soloact', 'data', 'config.yaml')
    }
    return SOURCES
