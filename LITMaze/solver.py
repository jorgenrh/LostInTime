
import cv2
import numpy as np 
import copy
import sys
sys.path.insert(0, 'trainer/')

from maze import Maze
from tile import Tile
from tilemodel import TileModel
from tileproperties import TileProperties


from matplotlib import pyplot as plt

class Solver():

	def __init__(self, image):
		self.tp = TileProperties()
		self.tm = TileModel()
		self.tm.load('tiles_svm.dat')
		#self.tm.load('tiles_svm.dat')
		self.maze = Maze(image)
		self.match_tiles()


	def match_tiles(self):
		descriptors = []
		for tile in self.maze.tiles.flatten():
			img = cv2.resize(tile.image, self.tm.sample_dim, interpolation=cv2.INTER_AREA) 
			#img = cv2.equalizeHist(img)
			descriptors.append(self.tm.hog.compute(img))

		result = self.tm.predict(np.array(descriptors))
		result = np.array(map(self.tp.get_label, map(int, result)))
		result = result.reshape([self.maze.rows, self.maze.cols])

		self.maze.set_tiles(result)

		self.pcount = np.zeros([self.maze.rows,self.maze.cols], dtype=int)


	def get_walls(self, row, col):
		top, right, bottom, left = 0,0,0,0
		if row-1 < 0: top = 1
		if col+1 >= self.maze.cols: right = 1
		if row+1 >= self.maze.rows: bottom = 1
		if col-1 < 0: left = 1
		return [top, right, bottom, left]

	def get_neighbors(self, row, col):
		top, right, bottom, left = Tile(None), Tile(None), Tile(None), Tile(None)
		if row-1 >= 0: top = self.maze.tiles[row-1,col]
		if col+1 < self.maze.cols: right = self.maze.tiles[row,col+1]
		if row+1 < self.maze.rows: bottom = self.maze.tiles[row+1,col]
		if col-1 >= 0: left = self.maze.tiles[row,col-1]
		return [top, right, bottom, left]

	def is_conn(self, conn1, conn2, pos=0):
		fconn = conn1[pos]
		idx = pos+2 if pos+2 < 4 else pos+2-4
		tconn = conn2[idx]
		if fconn & tconn: return 1
		return 0

	def get_blocked(self, row, col, incwalls=False):
		tile = self.maze.tiles[row,col]
		neighbors = self.get_neighbors(row, col)
		blocked = [0,0,0,0]
		#blocked = self.get_walls(row, col) if incwalls else [0,0,0,0]

		for pos, neighbor in enumerate(neighbors):
			if neighbor.label and len(neighbor.possible) < 2: # locked
				npos = pos+2 if pos+2 < 4 else pos+2-4 # get opposite side
				if neighbor.con[npos] == 0:
					blocked[pos] = 1

		return blocked

	def get_possible(self, row, col):
		tile = self.maze.tiles[row,col]
		if tile.locked: return []

		possible = []
		group = [Tile(l) for l in tile.group]
		walls = self.get_walls(row,col)
		if not 1 in walls: 
			possible = [g.label for g in group]
		else:
			for t in group:
				res = [a & b for a, b in zip(walls, t.con)]
				if res != walls: possible.append(t.label)

		#return possible

		impossible = []
		absolutes = []
		neighbors = self.get_neighbors(row,col)
		for n,neighbor in enumerate(neighbors):
			if neighbor.locked or not neighbor.label:
				for pos in possible:
					p = Tile(pos)
					is_conn = self.is_conn(p.con, neighbor.con, n)
					if p.con[n] and not is_conn:
						impossible.append(pos)
					elif is_conn:
						absolutes.append(pos)


		result = [pos for pos in possible if pos not in impossible]
		if len(absolutes) > 0: result = absolutes
		'''
		print 'for', tile.label, (row,col)
		print '  neighbors ->', [n.label for n in neighbors]
		print '  possible ->', possible
		print '  impossible ->', impossible
		print '  absolutes ->', absolutes
		print '  result ->', result
		'''		
		return result


	def check_possible(self):
		for row in xrange(self.maze.rows):
			for col in xrange(self.maze.cols):
				tile = self.maze.tiles[row,col]
				tile.possible = self.get_possible(row, col)
				if len(tile.possible) > 0:
					if len(tile.possible) == 1 or tile.label not in tile.possible:
						tile.set(tile.possible[0])
				self.pcount[row,col] = len(tile.possible)

	def get_neighbor(self, row, col, direction):

		nrow,ncol = row,col
		if direction == 0: nrow -= 1
		elif direction == 1: ncol += 1
		elif direction == 2: nrow += 1
		elif direction == 3: ncol -= 1

		if 0 <= nrow < s.maze.rows: row = nrow
		else: return Tile(None), row, col
		if 0 <= ncol < s.maze.cols: col = ncol
		else: return Tile(None), row, col

		return self.maze.tiles[row,col], row, col

	def get_pcount(self):
		for row in xrange(self.maze.rows):
			for col in xrange(self.maze.cols):
				tile = self.maze.tiles[row,col]
				self.pcount[row,col] = len(tile.possible)
		return self.pcount


	# ----------------------------------------------------------------------

	route = []


	

	def start_routing(self, corner=0):
		corners = [(0,0), (0,8), (5,8), (5,0)]
		row,col = corners[corner]
		label = (self.maze.tiles[row,col]).label

		self.route = []

		print 'self.search_route(%d, %d, %s)' % (row,col,label)
		result = self.search_route(row, col, label, -1)

		if result: return np.array(self.route)

		return False


	def route_undo(self, reverse=True):

		route = enumerate(self.route)
		if reverse: route = reversed(list(enumerate(self.route)))

		for index, data in route:		

			(row, col) = data[0]
			label = data[1]
			connected_side = data[2]
			possible = data[3]
			label_index = data[4]
			output_index = data[5]

			new_output_index = output_index # todo
			new_label_index = label_index

			changes_made = False

			# change T-output
			if label[0] == 'T' and output_index == 0:
				new_output_index = 1
				self.route = [d for n,d in enumerate(self.route) if n < index]
				#return row, col, label, connected_side, possible, label_index, new_output_index
				return self.route_search(row, col, label, connected_side, 
										 possible, label_index, new_output_index)

			# change label (rotation)
			elif len(possible) > 1 and label_index+1 < len(possible):
				new_label_index = label_index+1
				new_label = possible[new_label_index]
				new_output_index = 0

				self.route = [d for n,d in enumerate(self.route) if n < index]

				#return row, col, new_label, connected_side, possible, new_label_index, output_index
				return self.route_search(row, col, new_label, connected_side, 
										 possible, new_label_index, output_index)


		return False


	# 1. set tile
	# 2. find next best suited tile 
	def route_search(self, row, col, label, connected_side, possble=[], label_index=0, output_index=0):

		tile = self.maze.tiles[row,col]
		tile.set(label)

		self.route.append([
			(row, col),
			label,
			connected_side,
			possble,
			label_index,
			output_index
		])


		outputs = [] 
		for side, val in enumerate(tile.con):
			if val == 1 and side != connected_side:
				outputs.append(side)
		#outputs = list(reversed(outputs))

		direction = outputs[output_index]
		#direction = outputs[0]

		next_connected_side = direction+2 if direction+2 < 4 else direction+2-4

		'''
		sides = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT', 'NOWHERE']
		print tile.label, 'has outputs', [sides[o] for o in outputs] 
		print 'this is connected at', sides[connected_side]
		print 'next is connected at', sides[next_connected_side]
		#return
		'''

		next_tile, next_row, next_col = self.get_neighbor(row, col, direction)
		next_label = next_tile.label

		next_possible = []
		for pos in next_tile.possible:
			pos = Tile(pos)
			if self.is_conn(tile.con, pos.con, direction):
				next_possible.append(pos.label)

		if len(next_possible): 
			next_label = next_possible[0]
			#next_tile.set(next_label)


		sides = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT']
		#print tile.label, 'has outputs', [sides[o] for o in outputs] 
		#print tile.label, (row,col), 'going:', sides[direction]
		#print 'NEXT:', next_label, (next_row, next_col)


		coords = [data[0] for data in self.route]
		error = False

		is_conn = self.is_conn(tile.con, next_tile.con, direction)
		if (next_row, next_col) == (3,4) and is_conn: 
			#print 'Finished!'
			return True
		elif (next_row, next_col) in coords:
			error = 'already exits in route'
			error = True
		elif len(next_possible) == 0: 
			error = 'has no possible matches'
			error = True

		if error:
			#print 'Error:', (next_row, next_col), next_tile.label, error
			#return False
			return self.route_undo()

		#return 'next'
		return self.route_search(next_row, next_col, next_label, next_connected_side, next_possible)


	def get_route(self, corner):
		corners = [(0,0), (0,8), (5,8), (5,0)]
		row,col = corners[corner]
		label = (self.maze.tiles[row,col]).label

		self.route = []
		route = self.route_search(row, col, label, -1)

		if not route: return False

		return list(self.route)

	all_routes = []
	route_point_ids = []
	
	def route_save_point(self, row,col,label,conn,out_index):
		point_id = '%d%d%s%d%d' % (row,col,label,conn,out_index)
		if point_id not in self.route_point_ids:
			self.route_point_ids.append(point_id)
			return True

		return False


	def route_search_all(self, base_route):
		for index,br in enumerate(base_route):

			(row,col),label,conn,pos,pos_index,out_index = br

			if label[0] == 'T' and out_index == 0:
				#print br
				#print br[0], label, 'out:', out_index, 'to out: 1'
				self.route = [d for n,d in enumerate(base_route) if n < index]
				out_index = 1
				if self.route_save_point(row,col,label,conn,out_index):
					alt_route = self.route_search(row, col, label, conn, pos, pos_index, out_index)
					if alt_route:
						#self.route_save(list(s.route))
						self.all_routes.append(list(self.route))
						self.route_search_all(list(self.route))

			elif len(pos) > 1 and pos_index+1 < len(pos):
				self.route = [d for n,d in enumerate(base_route) if n < index]
				#print 'changing', (row,col), label, 'to',
				#print br[0], label, 'to', 
				pos_index += 1
				label = pos[pos_index]
				out_index = 0
				#print label
				if self.route_save_point(row,col,label,conn,out_index):
					alt_route = self.route_search(row, col, label, conn, pos, pos_index, out_index)
					if alt_route:  
						self.all_routes.append(list(self.route))
						self.route_search_all(list(self.route))


	def get_all_routes(self, corner):
		sides = ['top left', 'top right', 'bottom right', 'bottom left' ]
		self.route_point_ids = []
		route = self.get_route(corner)
		self.all_routes = [route]
		if route:
			print 'Mapping all routes for', sides[corner], 'corner'
			base_route = list(route)
			#print 'BASEROUTE:',base_route
			self.route_search_all(base_route)
			#print self.all_routes
			return list(self.all_routes)
		else:
			print 'Couldn\'t find base route for', sides[corner], 'corner' 
			return False




	def check_route_in_solution(self, solution, route):
		for point in route: 
			row,col = point[0]
			label = point[1]
			if solution[row,col] and solution[row,col] != label:
				return False
		return True

	def check_route_combo(self, routeA, routeB, routeC, routeD):
		solution = np.chararray([self.maze.rows,self.maze.cols], itemsize=2)
		solution[:] = ''

		routes = [routeA, routeB, routeC, routeD]
		for route in routes:
			for point in route:
				row,col = point[0]
				label = point[1]
				if solution[row,col] and solution[row,col] != label:
					return []
				else:
					solution[row,col] = label

		return solution

	def find_solution(self):

		all_routes = []
		for c in xrange(4):
			routes = s.get_all_routes(c)
			all_routes.append(routes) 
			print 'Corner', c, 'has', len(routes), 'routes'

		solutions = []

		for numA,routeA in enumerate(all_routes[0]):
			for numB,routeB in enumerate(all_routes[1]):
				for numC,routeC in enumerate(all_routes[2]):
					for numD,routeD in enumerate(all_routes[3]):
						solution = self.check_route_combo(routeA, routeB, routeC, routeD)
						if len(solution):
							solution = list(solution.flatten())
							for n,val in enumerate(solution):
								if not val: solution[n] = 'B0'
							solution = np.array(solution).reshape([self.maze.rows,self.maze.cols])
							solutions.append([numA,numB,numC,numD])
							print 'COMBO:', solutions[-1]
							return solution

		return solutions



#def get_all_routes()


if __name__ == '__main__':

	maze  = 'maze_images/maze11.png'
	sides = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT']
	dirs  = ['up', 'right', 'down', 'left']
	NOWHERE, TOP, RIGHT, DOWN, LEFT = -1,0,1,2,3
	
	s = Solver(maze)
	#plt.imshow(s.maze.get_maze()), plt.show(), exit()
	#if '7' in maze: (s.maze.tiles[3,4]).set('E0') #hack maze7
	#if '1' in maze: (s.maze.tiles[2,3]).set('C3') #hack maze1
	#if '8' in maze: (s.maze.tiles[2,3]).set('C3')
	#plt.imshow(s.maze.get_maze()), plt.show(), exit()
	s.check_possible()

	#routes = s.get_all_routes(2)
	#print len(routes)
	#plt.imshow(s.maze.get_maze()), plt.show(),exit()


	'''
	for c in xrange(4):
		routes = s.get_all_routes(2)
		print 'Corner', 2, 'has', len(routes), 'routes'

	exit()
	'''
	
	solution = s.find_solution()
	#solution = np.array(list(map(s.tp.get_num, list(solution.flatten()))))
	#solution = solution.reshape([s.maze.rows, s.maze.cols])
	#print solution
	if len(solution):
		print np.array(solution)
		s.maze.set_tiles(solution)
		plt.imshow(s.maze.get_maze(True,False,False)), plt.show()
	else:
		print 'NO SOLUTION FOUND!'


	exit()
