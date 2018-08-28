import numpy as np
import librosa
import h5py
import os
import sox

# filename = librosa.util.example_audio_file()
# y, sr = librosa.load(filename, sr = 41000, duration = 10)
#
#
#
# path = '/Users/BhavishDaswani/Desktop/DataScienceRetreat2018/Project/soundExploration/soloact/notebooks/'
# ff = h5py.File(path + 'training_sets.hdf5', 'r')
# data = np.array(ff['training'])
# print ('Shape : {}, {}'.format(*data.shape))
#
#
#
# def extract_features_means(parent_dir, sub_dirs, file_ext='*.wav'):
#     features, labels = np.empty((0, 193)), np.empty(0)
#     # for label, sub_dir in enumerate(sub_dirs):
#         print('parsing %s...' % sub_dir)
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             try:  # sometimes failss??
#                 # mean value of spectral content for Feed Forward Net
#                 X, sample_rate = librosa.load(fn)
#                 stft = np.abs(librosa.stft(X))
#                 mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#                 chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#                 mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
#                 contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
#                 tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
#             except:
#                 print('error, skipping...', fn)
#                 pass
#             ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
#             features = np.vstack([features, ext_features])
#             labels = np.append(labels, fn.split('/')[fn.count('/')].split('-')[1])
#     return np.array(features), one_hot_encode(np.array(labels, dtype=np.int))
#
#
#
# stft = np.abs(librosa.stft(X))
# mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
# chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
# mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
# contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
# tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
