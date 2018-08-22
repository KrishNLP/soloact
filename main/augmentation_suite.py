import librosa 
import numpy as np 
from inspect import signature
from flatten_json import flatten
import random
from pysndfx import AudioEffectsChain
import pandas as pd


def load_track(path, sr = 44100):
	x, sr = librosa.load(path, sr = sr) # default
	return x

# Note that anything changing frequency changes the shape of our input!

# effects = {'overdrive' : {'gain' : (0, 75)}}

def augment_track(path, exercise, augment_n, effects): 

	"""
	Procedurally augment track N times with sound effects, output determined by ML exercise

	AudioEffectsChain effects are externalised as callable standalones
	Effects dictionary is traversed and parameters accumulated through function call
	Parameter values are randomized between bounds provided unless..
	Dictionary contains request for 'default' values,'toggle' or None (0).
	Toggle being True or False values required
	Effects stored and returned with augmented track (labels, features)
	

	Parameters
	----------
	path : str
		Filepath
	exercise : str
		'Regresison' or 'Classification'
	augment_n: int
		Number of generated files from the one
	effects: dict of str, int and bools
		Extent to which effect is added

	Returns
	-------
	list of tuples
		track as np.array and features as python dictionary

	"""
	
	track = load_track(path)
	
	all_tracks = []

	for sample in range(augment_n): # number we're augmenting
		
		fx_manager = AudioEffectsChain() # Initialized with 0 features
		
		features = {}

		for effect, settings in effects.items():
		
			standalone_effect = getattr(fx_manager, effect)
			# return random range within max settings applied
			settings_applied = {}
			
			for parameter, value in settings.items():
				
				default_ = signature(standalone_effect).parameters[parameter].default

				if value == 'toggle': # for binary options
					settings_applied[parameter] = random.choice([True, False])
				else:
					# Apply default here
					if value == 'default':
						if exercise == 'classification':
							settings_applied[parameter] = random.choice([0, default_])
							continue       
						settings_applied[parameter] = default_
					else:

						if all([x is None for x in value]):
							settings_applied[parameter] = 0
							continue

						lower, upper = [default_ if v == 'default' else v for v in value]
						# At least have a max when giving settings!
						randomized_val = random.randint(lower, upper)
						if exercise == 'classification':
							settings_applied[parameter] = random.choice([0, randomized_val])
							continue
						settings_applied[parameter] = randomized_val

			# Give effects to fx_manager to compose pipeline
			standalone_effect(**settings_applied)                         

			# Validate added effects
			assert fx_manager.__dict__.get('command') != [], 'Settings not applied'

			# Record features

			features = {**{effect : settings_applied}, **features} 

		# Apply effects!
		augmented_track = fx_manager(track)

		# Since we're not modifying frequency input and output shape should match
		assert track.shape == augmented_track.shape, 'Output shape must match input'

		all_tracks.append((augmented_track, features))

	return all_tracks

def make_tabular(samples):
	"""
	Make tabular augmented tracks (NDarray) and features (DataFrame) 
	"""

	X, features = zip(*samples)
	# give each element 2 dimensions (1, len(vector)) - change this!
	X = np.atleast_2d(*X) 
	stacked = np.stack(X)
	# Using dot notation for column names after unnesting records
	unnest = lambda x: flatten(x, separator = '.')
	effects_added = pd.DataFrame(list(map(unnest, features)))
	return (stacked, effects_added)

	
def sound_factory(soundfiles, batch_size, **kwargs):

	"""
	Produce batches 

	
	"""
	
	while True:
		
		# take n samples from series object passed

		# print (soundfiles)
		batch_files = random.sample(soundfiles, k = batch_size)
		
		batch_inputs = []
		batch_outputs = []
		
		for soundfile in batch_files:    
			
			augmented_sound_files = augment_track(soundfile, **kwargs)  
			# features already stacked!
			features, labels = make_tabular(augmented_sound_files)
			
			if kwargs.get('exercise') == 'classification':
				# binarize each column
				labels = labels.where(labels == 0).fillna(1)
			
			# pandas to numpy structured array
			label_array = np.vstack(np.asarray(labels.to_records(index = False)))
			
			batch_inputs.append(features)
			batch_outputs.append(label_array)
		
		max_feature = max([i.shape for i in batch_inputs], key = lambda x: x[2])
	
		def padder(inp, max_shape): 
			# padding end with zeros matching maxfeature input shape
			zero_grid = np.zeros(max_shape)
			x,y,z = inp.shape
			zero_grid[:x, :y, :z] = inp
			return zero_grid
	
		# not sure if I have to keep stacking
		batch_x = np.vstack([padder(b, max_feature) for b in batch_inputs])
		# how do we represent outputs?
		batch_y = np.vstack(batch_outputs)
		
		yield (batch_x, batch_y)

# # SAMPLE EFFECTS
# effects = {'overdrive' : {'gain' : (None, 75), 'colour' : (None, 20)},
#                    'reverb' : {'wet_only' : 'toggle'}      
#                   }

# # DESIRED GENERATOR INPUT
# batch_settings = {
	
#     'effects' : effects,
#     'exercise' : 'classification',
#     'batch_size' : 2, # 30 files
#     'augment_n' : 20
	
# }


