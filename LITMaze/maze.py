
import cv2
import numpy as np 
import os.path

from tile import Tile

class Maze(object):

	screen = []

	def __init__(self, img_fn=''):
		self.cols = 9
		self.rows = 6
		self.tile_wh = (83,83)
		self.tile_box = (79,79) #(69,69)
		self.maze_xy = (290, 54)
		self.maze_wh = (self.cols * self.tile_wh[0], self.rows * self.tile_wh[1])

		self.tile_images = []
		self.tiles = []

		if img_fn: self.load(img_fn)


	def extract_tile_images_OLD(self):
		mh, mw = self.maze.shape[:2]
		tw, th = self.tile_wh
		tiles = np.array([np.hsplit(row, int(mw/tw)) for row in np.vsplit(self.maze, int(mh/th))])
		#tiles = tiles.reshape(-1, th, tw)
		return tiles

	def extract_tile_images(self):
		mh, mw = self.maze.shape[:2]
		tw, th = self.tile_wh
		tiles = np.array([np.hsplit(row, int(mw/tw)) for row in np.vsplit(self.maze, int(mh/th))])
		#tiles = tiles.reshape(-1, th, tw)

		px,py = int((self.tile_wh[0]-self.tile_box[0])/2), int((self.tile_wh[1]-self.tile_box[1])/2)
		bx,by = self.tile_box

		real_tiles = []
		for row in xrange(self.rows):
			tile_row = []
			for col in xrange(self.cols):
				tile = tiles[row,col]
				tile_row.append(np.array(tile[py:by, px:bx], dtype=np.uint8))
			real_tiles.append(tile_row)
		tiles = np.array(real_tiles)

		#plt.imshow(tile[py:by, px:bx], 'gray'), plt.show(), exit()
		return tiles

	def draw_grid(self, img, line_size=1, color=(0, 255, 0)):

		font, font_scale = cv2.FONT_HERSHEY_COMPLEX_SMALL, 1

		mx, my = self.maze_xy
		mw, mh = self.maze_wh
		tw, th = self.tile_wh

		cv2.rectangle(img, (mx,my), (mx+mw, my+mh), color, line_size)
		for n in xrange(0, self.cols):
			px = mx + n * tw - 1
			label = str(n)
			lw,lh = cv2.getTextSize(label, font, font_scale, 0)[0]
			cv2.putText(img, label, (px+int(tw/2-lw/2), my-int(lh/2)), font, font_scale, color)
			if n > 0: cv2.line(img, (px, my), (px, my+mh), color, line_size)

		for n in xrange(0, self.rows):
			py = my + n * th - 1
			label = str(n)
			lw,lh = cv2.getTextSize(label, font, font_scale, 0)[0]
			cv2.putText(img, label, (mx-(lw+lw/2), py+(th/2+lh/2)), font, font_scale, color)
			if n > 0: cv2.line(img, (mx, py), (mx+mw, py), color, line_size)

		return img

	def draw_labels(self, img):
		
		color = (0,0,0)
		font, font_scale = cv2.FONT_HERSHEY_COMPLEX_SMALL, 1

		mx, my = self.maze_xy
		tw, th = self.tile_wh

		for row in xrange(self.rows):
			for col in xrange(self.cols):
				label = self.tiles[row,col].label
				lw,lh = cv2.getTextSize(label, font, font_scale, 0)[0]
				lx, ly = mx + col * tw, my + row * th 
				cv2.putText(img, label, (lx+1, ly+lh+4), font, font_scale, color)

		return img

	def draw_shapes(self, img):

		color = (255,0,0)
		line_size = 5

		mx, my = self.maze_xy
		tw, th = self.tile_wh

		for row in xrange(self.rows):
			for col in xrange(self.cols):
				if self.tiles[row,col].label:
					pattern = self.tiles[row,col].shape
					bx = mx + col * tw
					by = my + row * th
					for p in pattern:
						ps = (int(bx+tw*p[0][0]), int(by+tw*p[0][1]))
						pe = (int(bx+th*p[1][0]), int(by+th*p[1][1]))
						cv2.line(img, ps, pe, color, line_size)

		return img

	def get_maze(self, cimg=True, grid=True, labels=True, shapes=True):
		img = self.screen.copy() if cimg else self.screen_gray.copy()

		if grid:   img = self.draw_grid(img)
		if labels: img = self.draw_labels(img)
		if shapes: img = self.draw_shapes(img)
		
		return img


	def set_tiles(self, tiles=[]):

		self.tile_images = self.extract_tile_images()
		self.tiles = []

		for row in xrange(self.rows):
			for col in xrange(self.cols):
				label = tiles[row,col] if len(tiles) else None
				tile = Tile(label) 
				tile.image = self.tile_images[row,col]
				self.tiles.append(tile)

		self.tiles = np.array(self.tiles).reshape([self.rows, self.cols])


	def load(self, img_fn):

		if not os.path.isfile(img_fn):
			exit('Error: %s does not exist' % img_fn) 

		self.screen = cv2.imread(img_fn)
		self.screen_gray = cv2.imread(img_fn, 0)

		mx, my = self.maze_xy
		mw, mh = self.maze_wh
		self.maze = self.screen_gray[my:my+mh, mx:mx+mw]

		self.set_tiles()




if __name__ == '__main__':

	from matplotlib import pyplot as plt

	file = 'maze_images/maze6.png'
	m = Maze(file)

	row,col = 0,0
	tile = m.tiles[row,col]

	plt.imshow(tile.image, 'gray'), plt.title(tile.label), plt.show()
