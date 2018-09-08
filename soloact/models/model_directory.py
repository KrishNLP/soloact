import os

base_models = {
    "multi_class_model(4)" : {
            'filepaths': {
            'config' : '1_config.yaml',
            'labels' :  '1_labels.csv',
            'features' : '1_features.npy'},
        'args' : {
            'predicted_effects' : ["overdrive.gain_db", "reverb.reverberance", "chorus.delays", "phaser.delay"],
            'scale' : False}
        },
    "joint_model" : {
            'filepaths': {
            'config' : '2_config.yaml',
            'labels' :  '2_labels.csv',
            'features' : '2_features.npy'},
        'args' : {
            'predicted_effects' : ["overdrive.gain_db", "reverb.reverberance", "chorus.delays", "phaser.delay"],
            'scale' : True}
        },
    "multi_class_model(3)" : {
            'filepaths': {
            'config' : '1_config.yaml',
            'labels' :  '1_labels.csv',
            'features' : '1_features.npy'},
        'args' : {
            'predicted_effects' : ["overdrive.gain_db", "reverb.reverberance", "chorus.delays"],
            'scale' : False}
        },
}
