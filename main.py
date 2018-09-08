import soloact
from soloact import write_annotation, write_chords, augment_data, dataset, joint_model, model_directory, multi_class_model
import os
import glob
import argparse
from pprint import pprint
from argparse import RawTextHelpFormatter

paths = soloact.make_source_paths(os.getcwd())
pitch_classes = list(paths.get('strategies').keys())


def w_annotations(args):

    """Dumb method"""

    write_annotation(paths)

def verify_chord(chord_type):
    """
    Chord generation verified before creation

    """
    chord_type = chord_type.pf
    if chord_type in pitch_classes:
        annotations = paths.get('interim').trace + '/meta.csv'
        write_chords(paths, annotations=annotations, write=True, strategy=chord_type)
    else:
        print ('Strategy "{}" not available'.format(chord_type))
        print ('Try one of these {}'.format(tuple(pitch_classes)))

def augment(args):
    features, labels = augment_data(paths = paths, configuration = args.cf,
    exercise = args.ex, augment_x = args.x,
    strategy = args.pf, feature_extraction = args.fx,
    subsample = args.s,
    write_augmented = args.wa, write_data = args.o)
    return features, labels

base_models = model_directory.base_models
def run_base_model(args):

    """
    Run packaged models

    """

    model_select = base_models.get(args.s)
    models_dir = paths.get('models_path')
    model_select['filepaths'] =  {k:models_dir()  + '/' + v for k,v in model_select.get('filepaths').items()}
    model_kwargs = {**model_select['filepaths'], **model_select['args']}
    if args.p == True:
        pprint (model_kwargs)
    if args.t == True:
        kwargs = dict(args._get_kwargs())
        hyperparameters = {k:v for k,v in kwargs.items() if k in ['batch_size', 'lr']}
        data = dataset.Dataset(**model_kwargs)
        data = data.make_dataset()
        if '(' in args.s:
            func = globals()[args.s.split('(')[0]]
        else:
            func = globals()[args.s]
        results = paths.get('reports_path')
        func.main(data, results_folder = results(), write_results = True if args.save == True else False, **hyperparameters)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # MAKE ANNOTATIONS
    make_meta_parser = subparsers.add_parser('make_meta', help =  "Create annotations file from raw, anchoring dependencies")
    make_meta_parser.add_argument('-r', help = 'creat meta file', nargs = '?')
    make_meta_parser.set_defaults(func = w_annotations)

    # Chord generation parser
    pitch_form_str = """

        - Chord creation function supporting triads, powerchords and septchords.

        - Files are written to "root/data/interim" as .wavs

        - A chord is any combination of three
        or more pitch classes that sound simultaneously.

    """
    chord_generation_parser = subparsers.add_parser('write_chords', help = pitch_form_str, formatter_class = RawTextHelpFormatter)
    chord_generation_parser.add_argument(
            '-pf', nargs = '?' , default = 'powers',
            type = str,
            help = 'Available chord strategies: {}'.format(pitch_classes))
    chord_generation_parser.set_defaults(func = verify_chord)

    # Augmentation parser
    config_str = """.yaml file of active effects chain and
    those to sustain during on/off random sequencing for classsification exercises
    see "soloact/data/configurations/config.yaml" for config used in default run"""

    augmentation_parser = subparsers.add_parser('augment_data',
            help = 'Duplicate tracks for randomized effects, return features (array) and labels (dataframe)')
    augmentation_parser.add_argument(
            '-cf', nargs = '?', metavar = 'yaml configuration path',
            help = config_str)
    augmentation_parser.add_argument(
            '-ex', default = 'classification', type = str, metavar = 'exercise',
            help = 'ML exercise to calibrate for classification induces effect on/off randomization')
    augmentation_parser.add_argument('-x', default = 5, type = int, metavar = 'augmentations/file',
            help = 'Number of times track is augmmented from original state and multiplied')
    augmentation_parser.add_argument('pf', default = 'powers',
                type = str, help = 'Chord kind to augment')
    augmentation_parser.add_argument('-fx', default = 'mfcc',
                            metavar = 'Feature extraction',
                            help = 'Feature extraction methods supported: [chroma, mfcc]')
    augmentation_parser.add_argument('-s', nargs = '?', type = int, metavar = 'sample',
                help = 'Use subset of tracks - suggested to couple with write augmented parameter')
    augmentation_parser.add_argument('-wa', action = 'store_true',
                help = """STORAGE CONCERNS: Write tracks to "root/data/interim/<chord>/augmented" matching number of
                augmentations requested by number of files (subsampling taken into account)""")
    augmentation_parser.add_argument('-o', action = 'store_true',
            help = 'Write features (npz) and labels (csv) to reports')
    augmentation_parser.set_defaults(func = augment)

    calibration_parser = subparsers.add_parser('train_existing',
        help  = 'Use pre-packaged configurations for training')

    calibration_parser.add_argument('s', default = 'multi_class_model', type = str,
            help = 'Select from packaged example. Available: {}'.format(list(base_models.keys()))
    )
    calibration_parser.add_argument(
                '-t', action = 'store_true', help = 'Train from preset')

    calibration_parser.add_argument('-p', action = 'store_true',
                    help = 'Pretty print config'
    )
    # offering redirect later
    calibration_parser.add_argument('-save', help = 'save weights and model history', action = 'store_true')
    calibration_parser.add_argument('-batch_size', help='batch_size', default = 32, nargs = '?', type = int)
    calibration_parser.add_argument('-lr', help = 'learning rate', default = 0.0003, nargs = '?', type = float)

    calibration_parser.set_defaults(func = run_base_model)

    args = parser.parse_args()
    args.func(args)
