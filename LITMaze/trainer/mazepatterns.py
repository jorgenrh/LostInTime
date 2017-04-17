
import cv2
import numpy as np

import sys
sys.path.insert(0, '../')

from tile import Tile



class MazePatterns():

	labels = [ 
		'B0', 
		'S0', 'S1', 'S2', 'S3', 
		'C0', 'C1', 'C2', 'C3',
		'L0', 'L1',
		'T0', 'T1', 'T2', 'T3', 
		'E0', 'E1',
		'A0', 'A1', 'A2', 'A3' 
	]

	B0 = 0
	S0, S1, S2, S3 = 1, 2, 3, 4
	C0, C1, C2, C3 = 5, 6, 7, 8
	L0, L1 = 9, 10
	T0, T1, T2, T3 = 11, 12, 13, 14
	E0, E1 = 15, 16
	A0, A1, A2, A3 = 17, 18, 19, 20	

	patterns = {
		'maze1.png': [
			S1, L0, L1, L1, C1, C2, C1, C3, S3,
			C0, C2, C3, C3, C0, C0, L1, C2, C3,
			L1, L0, L1, C2, L1, L1, T2, C2, L0,
			L0, T3, C1, C3, E1, L1, C0, L0, L1,
			L0, C0, C0, C3, C3, C3, C1, C1, C0,
			S0, B0, C1, L1, C2, C0, L1, L1, S3
		],
		'maze2.png': [
			S1, C1, C2, L0, C1, C2, L0, C1, S2,
			L0, C0, C3, C0, C0, C3, C1, C0, C3,
			C1, C2, L0, T1, C2, L0, C1, C2, C3,
			C0, C3, C0, C3, E0, L0, C0, C3, L1,
			C1, C2, L1, L1, T1, C2, L0, C1, C2,
			S1, C3, C0, L1, C3, C0, C3, C0, S3
		],
		'maze3.png': [
			S1, C3, C0, L0, L0, L0, C2, C3, S3,
			C2, C3, L1, C3, L0, L0, T0, C3, C1,
			C3, L1, C0, L1, C2, C3, C1, C1, L0,
			C2, L1, C2, T0, A3, C1, C1, L0, L1,
			L0, C1, C0, C3, C3, C1, C2, C2, C3,
			S0, C1, L0, L0, C1, C3, L0, L0, S3
		],
		'maze4.png': [
			S1, L1, L1, C0, C0, L1, L1, C3, S2,
			C1, L0, L0, C1, L0, C2, L1, C2, L1,
			C3, L0, L0, L0, C1, C1, T1, L0, C2,
			C3, L0, L0, L1, E1, L1, T2, C1, C2,
			C0, L0, L1, L1, C0, C1, C0, L0, L0,
			S1, L0, L0, L1, C1, C3, L0, C3, S0
		],
		'maze5.png': [
			S1, L0, L1, L1, C3, L0, C1, L1, S3,
			C0, L1, L1, C2, L0, C1, C0, C3, C0,
			L0, C2, C3, C3, C1, L0, C2, C3, L0,
			T2, C1, C0, L1, A2, L1, L0, C0, T1,
			L0, C1, L1, L0, C2, C1, C3, L0, L0,
			S0, C3, L0, L1, L0, L1, L1, C0, S0
		],
		'maze6.png': [
			S1, L1, T3, L1, L0, L0, L0, C0, S2,
			C1, L0, C0, C2, L0, C3, C0, C2, L1,
			L1, C1, C2, C1, C2, L1, T3, L1, C3,
			L1, L0, C3, C3, E0, C1, C2, C1, C1,
			C3, C1, C0, C0, L1, C3, C1, L0, L1,
			S1, L0, C2, B0, C3, C3, C3, C0, S0
		],
		'maze7.png': [
			S1, L0, C1, L1, C2, C1, C2, L0, S3, 
			C3, L0, C0, C2, C0, C2, T2, L0, C0, 
			C2, L0, L0, C0, C2, T3, C0, C2, C1, 
			C3, L0, C0, C2, E0, L1, L1, C3, C3, 
			L1, C0, C2, C0, C2, L1, C2, L0, C1, 
			S0, C2, L0, C1, L0, C2, L0, L0, S3
		],
		'maze8.png': [
			S1, L0, L1, L1, L1, L1, L1, C3, S2, 
			C2, L0, L0, T0, L1, C3, C1, T0, C0, 
			C3, C1, C2, C3, B0, L0, C0, L0, C1, 
			C2, C1, L1, C1, A2, C3, L0, L1, C1, 
			C2, C3, L1, L0, L1, C2, C0, C0, C3, 
			S1, C1, C1, C3, C0, C2, C2, C2, S0 
		],
		'maze9.png': [
			S2, C3, L0, C3, L1, C0, C2, B0, S2, 
			L1, C1, C1, L1, C1, C0, C1, L0, C1, 
			C3, L0, C3, L1, C0, L0, C0, C2, C2, 
			C3, L0, T1, C1, A1, L0, C2, L1, L1, 
			C0, C0, C2, C0, C3, L1, T2, C1, L0, 
			S1, C2, C3, C2, L1, L1, C0, B0, S0 
		],
		'maze10.png': [
			S2, L1, C3, L0, C3, C2, L1, L1, S3, 
			C3, C0, C1, C0, L0, L1, C2, C2, C2, 
			C0, T0, C3, C2, L0, C3, C2, C3, C1, 
			L1, L1, C2, C0, E0, C2, L0, L0, C1, 
			L0, L1, C0, C3, L0, T0, C2, C0, C2, 
			S0, C1, C1, B0, C0, C3, C3, C0, S0
		],
		'maze11.png': [
			S2, L1, C1, L0, C2, C3, L0, C2, S2, 
			L1, C2, C1, C0, C2, L1, L1, C2, C0, 
			C3, C0, C3, C3, B0, L1, C0, L1, C2, 
			C0, L1, C0, C2, A2, T0, C0, C3, C1, 
			T1, C2, L0, L1, L1, L1, C1, C1, L0, 
			S0, C0, L0, C1, C2, C3, C2, L0, S3 
		]
	}

	def get_tiles(self, maze):
		if maze in self.patterns:
			pattern = np.array([self.labels[n] for n in self.patterns[maze]])
			pattern = pattern.reshape([6,9])
			return pattern
		else:
			exit('Error: %s does not have a pattern' % maze)

	def get_files(self):
		return list(sorted(self.patterns.keys()))


	def guess_tiles(self, image):
		from maze import Maze
		from tilemodel import TileModel
		from tileproperties import TileProperties
		from matplotlib import pyplot as plt
		from os import path 

		tp = TileProperties()
		maze = Maze(image)

		tm = TileModel()
		tm.load('../tiles_svm.dat')

		descriptors = []
		for tile in maze.tiles.flatten():
			img = cv2.resize(tile.image, tm.sample_dim, interpolation=cv2.INTER_AREA) 
			descriptors.append(tm.hog.compute(img))

		result = tm.predict(np.array(descriptors))
		result = np.array(map(tp.get_label, map(int, result)))
		result = result.reshape([maze.rows, maze.cols])

		maze.set_tiles(result)

		filename = path.basename(image)
		print '\'%s\': [' % filename
		for row in xrange(maze.rows):
			line = '\t'
			for col in xrange(maze.cols):
				line += maze.tiles[row,col].label + ', '
			print line
		print ']'

		plt.imshow(maze.get_maze()), plt.show()

	def test_tiles(self, image):
		from maze import Maze
		from tileproperties import TileProperties
		from matplotlib import pyplot as plt

		tp = TileProperties()

		img = '../maze_images/' + image
		maze = Maze(img)

		tiles = np.array(map(tp.get_label, map(int, self.patterns[image])))
		tiles = tiles.reshape([maze.rows, maze.cols])

		maze.set_tiles(tiles)

		plt.imshow(maze.get_maze()), plt.show()


if __name__ == '__main__':
	
	mp = MazePatterns()


	mp.guess_tiles('../maze_images/maze11.png')
	#mp.test_tiles('maze10.png')