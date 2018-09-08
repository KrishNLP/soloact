"""

Joint classificatio model

"""
from keras import backend as K
import keras
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Input, Flatten, Dropout, Activation,  Dense, Embedding
from keras.callbacks import ModelCheckpoint
import pickle
from .utils import dump_pickle

def main(data, results_folder, lr = 0.0001, batch_size = 32, write_results = False):

    K.clear_session()

    X_train = data['X_train']
    y_train = data['y_train']
    y_train_bin = data['y_train_bin']

    X_val = data['X_val']
    y_val = data['y_val']
    y_val_bin = data['y_val_bin']

    train_states = data['y_train_states']
    test_states = data['y_val_states']

    inputs = Input(shape = X_train.shape[1:])

    shared = Conv1D(128, 5, padding='same', activation = 'relu')(inputs)
    shared = Conv1D(128, 5, padding='same', activation = 'relu')(shared)
    shared = Dropout(0,1)(shared)
    shared = MaxPooling1D(pool_size=8)(shared)
    shared = Conv1D(128, 5, padding='same', activation = 'relu')(shared)
    shared = Conv1D(128, 5, padding='same', activation = 'relu')(shared)
    shared = Conv1D(128, 5, padding='same', activation = 'relu')(shared)
    shared = Dropout(0.3)(shared)
    shared = Conv1D(128, 5, padding='same', activation = 'relu')(shared)
    shared = Flatten()(shared)

    classifier = Dense(len(train_states), activation='softmax')(shared)
    classifier = Dense(len(train_states), activation = 'softmax', name='classifier')(classifier)

    regressor = Dense(1, activation='relu')(shared)
    regressor = Dense(1, activation='relu', name = 'regressor')(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor]) # regressor
    optimizer = keras.optimizers.Adam(lr=lr)

    model.compile(
                optimizer=optimizer,
                loss = {
                    'regressor' : 'mae',
                    'classifier': 'categorical_crossentropy'},
                metrics = {'classifier' : 'accuracy', 'regressor' : 'mae'}
    )
    # Harcoding regression model column index
    history = model.fit(X_train,
            {'classifier' : y_train_bin, 'regressor' : y_train[:, :1]},
            batch_size=batch_size,
            epochs=200,
            validation_data=(X_val, {'classifier' : y_val_bin, 'regressor' : y_val[:, :1]}))

    if write_results:
        # appends datetime to name
        dump_pickle(obj = history.history, name = results_folder + '/histories' )
