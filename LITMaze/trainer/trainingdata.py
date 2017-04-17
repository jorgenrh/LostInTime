
import cv2
import numpy as np
import itertools as it
import copy

import sys
sys.path.insert(0, '../')

from maze import Maze
from tile import Tile
from mazepatterns import MazePatterns
from tileproperties import TileProperties
from tilemodel import TileModel

class TrainingData():

	files = []
	file_path = '../maze_images/'

	num_samples = 25
	sample_dim = (20,20)


	def __init__(self):
		self.mazes = []
		self.tiles = {}
		self.sets = {}

		self.mp = MazePatterns()
		self.files = self.mp.get_files()
		self.tp = TileProperties()
		self.tm = TileModel()
		self.sample_dim = self.tm.sample_dim

		self.extract_files()


	def extract_files(self):

		for file in self.files:
			maze = Maze(self.file_path+file)
			maze.set_tiles(self.mp.get_tiles(file))
			self.mazes.append(maze)

		for label in self.mp.labels: self.tiles[label] = []

		for maze in self.mazes:
			tiles = maze.tiles.flatten()
			for tile in tiles: self.tiles[tile.label].append(tile)


	def convert_tile(self, tile, to_label):

		new_tile = Tile(to_label)
		angle = [0, 90, 180, 270][tile.pos-new_tile.pos]

		if angle != 0:
			center = tuple(np.array(tile.image.shape)[:2]/2) 
			rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
			new_tile.image = cv2.warpAffine(tile.image, rot_mat, tuple(np.array(tile.image.shape)[:2]), flags=cv2.INTER_LINEAR)
			return new_tile

		return tile

	def gamma_correction(self, img, correction):
		img = img/255.0
		img = cv2.pow(img, correction)
		return np.uint8(img*255)

	def distort_tile(self, tile):

		img = tile.image.copy()

		con = tile.con
		 # L0, L1 (E0, E1)
		 # T1, T3
		 # S0, S2
		if np.random.randint(0,2) > 0:
			if (con == [1,0,1,0] or con == [0,1,0,1] or 
				con == [0,1,1,1] or con == [1,1,0,1] or
				con == [1,0,0,0] or con == [0,0,1,0]):
				img = cv2.flip(img, 1)
			#print tile.label, 'horizontal flip'

		# L0, L1 (E0, E1)
		# T0, T2
		# S1, S3
		if np.random.randint(0,2) > 0:
			if (con == [1,0,1,0] or con == [0,1,0,1] or	
				con == [1,1,1,0] or con == [1,0,1,1] or 
				con == [0,1,0,0] or con == [0,0,0,1]):   
				img = cv2.flip(img, 0)
			#print tile.label, 'vertical flip'


		method = np.random.randint(0,3)
		if method == 0: cv2.fastNlMeansDenoising(img, img, np.random.randint(11,27), 7, 21) 
		if method == 1: img = cv2.medianBlur(img, [3,5,7,9][np.random.randint(0,4)])

		method = np.random.randint(0,3)
		if method == 0: img = self.gamma_correction(img, 0.5)
		if method == 1: img = self.gamma_correction(img, 1.5)

		tile.image = img
		return tile


	def create_sets(self, num_samples=None, sample_dim=None):

		num = num_samples if num_samples else self.num_samples
		dim = sample_dim if sample_dim else self.sample_dim

		print 'Creating %d sets of %d samples ...' % (len(self.tiles), num)

		self.sets = {}
		i = 0
		for label, images in self.tiles.iteritems():

			tiles = self.tiles[label]
			samples = tiles[:num]

			print '%3d%% - Generating %d samples for %s ...' % (np.round(100*i/(len(self.tiles)-1)), num-len(samples), label)

			if len(samples) < num:
				group = self.tp.get_group(label)
				if len(group) > 1: group = [l for l in group if l != label]

				group_samples = []
				for glabel in group:
					for gtile in self.tiles[glabel]:
						group_samples.append(self.convert_tile(gtile, label))
				samples += group_samples

				tmp_samples = list(samples)
				while len(samples) < num: 
					samples += [self.distort_tile(copy.copy(t)) for t in copy.copy(tmp_samples)]

			result = [cv2.resize(t.image, dim, interpolation=cv2.INTER_AREA) for t in samples[0:num]]
			#result = [cv2.equalizeHist(timg) for timg in result]
			self.sets[label] = result

			i += 1

		print '%d samples created (%dx%d)' % (len(self.tiles)*num, dim[0], dim[1])

	def get_sheet(self, label):
		if label not in self.sets: return np.zeros(self.sample_dim)

		cols = int(np.ceil(self.num_samples / float(np.sqrt(self.num_samples))));
		samples = iter(self.sets[label])

		s0 = samples.next()
		pad = np.zeros_like(s0)
		samples = it.chain([s0], samples)
		args = [iter(samples)] * cols
		rows = it.izip_longest(fillvalue=pad, *args)
		
		return np.vstack(map(np.hstack, rows))

	def get_training_data(self):
		if not len(self.sets): exit('Error: no training sets created')
		samples = []
		labels = []
		for label, data in self.sets.iteritems():
			samples += data
			labels += [self.tp.get_num(label)]*len(data) 
		return np.array(samples), np.array(labels)


if __name__ == '__main__':

	from matplotlib import pyplot as plt

	td = TrainingData()

	td.create_sets(50)

	t = td.get_training_data()

	sheet = td.get_sheet('T0')
	plt.imshow(sheet, 'gray'), plt.show()






