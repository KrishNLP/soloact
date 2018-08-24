effects = {
    'overdrive' :  {'gain' : (10,80), 'colour' :  (0,20)}, # these are really the only two we randomize
    'reverb': {
            'reverberance' : (0,100),
            'hf_damping': 50,
            'pre_delay': 20,
            'reverberance': 50,
            'room_scale': 100,
            'stereo_depth': 100,
            'wet_gain': 0,
            'wet_only': False
                },
    'delay' :
            {'decays': [0.3, 0.25],
             'delays': [1000, 1800],
             'gain_in': 0.8,
             'gain_out': 0.5,
             'parallel': False}
}

# Which effects we want randomized if regression
effects_to_randomize = {'overdrive': ['gain', 'colour']}
# Which should remain constant if classification
effects_to_steady = {'overdrive': ['gain']} # 