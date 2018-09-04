import soloact
from soloact import write_annotation, write_chords, augment_data

import os
if __name__ == '__main__':

    paths = soloact.make_source_paths(os.getcwd())

    write_chords_flag = True
    if write_chords_flag:
        annot = write_annotation()

        write_chords(write=True, source='power')

    X_train, y_train = augment_data(
        source='power',
        make_training_set=True,
        n_augment=1,
        SOURCES=paths
    )

