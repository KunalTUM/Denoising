import numpy as np
import scipy.io as sio
import glob, os
import tensorflow as tf
from imageio import imread
from tensorflow.keras.utils import Sequence
from utils import normalize, ffdnet_struct

class DataGenerator(Sequence):

	def __init__(self, 
				noisy_path,
				gt_path,
				batch_size=128,
				shuffle=True, 
				patch_size=(150, 150, 1),		#(50, 50, 3),
				input_size=(75,75,5)):			#(25, 25, 15)):

		# Initializations
		super(DataGenerator, self).__init__()
		self.noisy_path = noisy_path
		self.gt_path = gt_path
		self.list_IDs = os.listdir(self.noisy_path)
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.patch_size = patch_size
		self.input_size = input_size
		self.on_epoch_end()


	def __len__(self):
		return int(np.floor(len(self.list_IDs) / self.batch_size))


	def __getitem__(self, index):

		# Generates indexes of the batched data
		indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

		# Get list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)
		return X, y


	def on_epoch_end(self):

		self.indexes = np.arange(len(self.list_IDs))

		if(self.shuffle):
			np.random.shuffle(self.indexes)


	def __data_generation(self, list_IDs_temp):
		# Initialization
		X = np.empty((self.batch_size, *self.input_size))
		y = np.empty((self.batch_size, *self.patch_size))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			X[i, ] = ffdnet_struct(normalize(np.load(os.path.join(self.noisy_path, ID))))

			# Store class
			loaded_array = np.load(os.path.join(self.gt_path, ID))
			y[i, ] = normalize(np.expand_dims(loaded_array, axis=-1))		# .replace('_noisy.png', '_gt.png')

			# poor design LMAO
		return X, y
