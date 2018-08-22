from pysndfx import AudioEffectsChain
import random
import numpy as np
from inspect import signature

# Sample 

effects = {'overdrive' : {'gain' : (0, 75)}}


class fxManager:

	def __init__(self, effects, exercise):
		self.state = AudioEffectsChain()
		self.effects = effects
		self.exercise = exercise
		self.applied = {}

	def apply_effects(self):

		for effect, settings in self.effects.items():

			standalone_effect = getattr(self.state, effect)
			# return random range within max settings applied
			settings_applied = {}
			
			for parameter, value in settings.items():
				
				default_ = signature(standalone_effect).parameters[parameter].default
				
				if value == 'toggle': # binary outputs, exercise doesn't matter
					settings_applied[parameter] = random.choice([True, False])
				
				else:

					if value == 'default':
						if self.exercise == 'classification':
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
						if self.exercise == 'classification':
							settings_applied[parameter] = random.choice([0, randomized_val])
							continue
						settings_applied[parameter] = randomized_val

			standalone_effect(**settings_applied)
			self.applied[effect] = settings_applied

class Track:


	def __init__(self, sr, number_augmented, effects):

		self.track_array = np.arange(100000)
		self.sr = sr
		self.track_length = '{:.2f} seconds'.format(self.track_array.size / sr)
		self.number_augmented = number_augmented
		self.desired_effects = effects
		self.effects_applied = {}
		self.samples = []


