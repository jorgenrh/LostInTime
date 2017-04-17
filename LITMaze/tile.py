
from tileproperties import TileProperties

class Tile(TileProperties):

	def __init__(self, label=None):
		self.set(label)
		self.image = None
		self.possible = []
		self.solution = None

	def set(self, label):
		if label:
			self.label  = label
			self.pos    = self.get_pos(label) 
			self.num    = self.get_num(label)
			self.locked = self.is_locked(label)
			self.group  = self.get_group(label)
			self.con    = self.get_connections(label)
			self.shape  = self.get_shape(label)
		else:
			self.label = self.pos = self.num = self.locked = self.group = self.shape = None
			self.con = [0,0,0,0]

		self.props = [self.label, self.pos, self.num, self.locked, self.group, self.con, self.shape]



if __name__ == '__main__':
	tile = Tile('L1')
	print tile.props
