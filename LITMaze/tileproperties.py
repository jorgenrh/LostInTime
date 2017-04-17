

class TileProperties():

	B0 = 0
	S0, S1, S2, S3 = 1, 2, 3, 4
	C0, C1, C2, C3 = 5, 6, 7, 8
	L0, L1 = 9, 10
	T0, T1, T2, T3 = 11, 12, 13, 14
	A0, A1, A2, A3 = 15, 16, 17, 18	
	E0, E1 = 19, 20

	labels = [ 
		'B0', 
		'S0', 'S1', 'S2', 'S3', 
		'C0', 'C1', 'C2', 'C3',
		'L0', 'L1',
		'T0', 'T1', 'T2', 'T3', 
		'E0', 'E1',
		'A0', 'A1', 'A2', 'A3' 
	]

	shapes = {
		'C': [
			[[(0.5,0), (0.5,0.5)], [(0.5,0.5), (1,0.5)]],
			[[(0.5,0.5), (0.5,1)], [(0.5,0.5), (1,0.5)]],
			[[(0.5,0.5), (0.5,1)], [(0,0.5), (0.5,0.5)]],
			[[(0.5,0), (0.5,0.5)], [(0,0.5), (0.5,0.5)]]
		],
		'L': [
			[[(0.5,0), (0.5,1)]],
			[[(0,0.5), (1,0.5)]] 
		],
		'T': [
			[[(0.5,0), (0.5,1)], [(0.5,0.5), (1,0.5)]],
			[[(0,0.5), (1,0.5)], [(0.5,0.5), (0.5,1)]],
			[[(0.5,0), (0.5,1)], [(0,0.5), (0.5,0.5)]],
			[[(0,0.5), (1,0.5)], [(0.5,0), (0.5,0.5)]]
		]
	}

	connections = {
		#     T R B L
		'B': [0,0,0,0],
		'S': [1,0,0,0],
		'C': [1,1,0,0],
		'L': [1,0,1,0],
		'T': [1,1,1,0],
		'E': [1,0,1,0],
		'A': [1,1,0,0]
	}



	def __init__(self):
		pass

	def get_group(self, label):
		group = [l for l in self.labels if l[0] == label[0]]
		return group

	def get_num(self, label):
		num = None
		if label in self.labels: num = self.labels.index(label)
		return num

	def get_label(self, num):
		label = None
		if 0 <= num < len(self.labels): label = self.labels[num]
		return label

	def get_pos(self, label):
		return int(label[1])

	def get_shape(self, label):
		shape = []
		l, n = label[:2]
		if l in self.shapes and 0 <= int(n) < len(self.shapes[l]):
			shape = self.shapes[l][int(n)]
		return shape

	def get_connections(self, label):
		con = [0,0,0,0]
		l, n = label[:2]
		if l in self.connections:		
			n = int(n)
			con = self.connections[l][-n:] + self.connections[l][:-n]
		return con

	def is_locked(self, label):
		l = label[0]
		if l in self.shapes: return False
		return True

if __name__ == '__main__':

	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

	tp = TileProperties()


