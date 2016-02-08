from enum import Enum
from math import sqrt

class UnionFind:

	def __init__ (self, xs=[]):
		self._parent = {}
		self._rank = {}
		for x in xs:
			self.add(x)

	def add (self, x):
		if x not in self._parent:
			self._parent[x] = x
			self._rank[x] = 0

	def find (self, x):
		'''Running time: O(log n)'''
		rx = x
		while self._parent[rx] != rx:
			rx = self._parent[rx]
		while self._parent[x] != x:
			next_x = self._parent[x]
			self._parent[x] = rx
			x = next_x
		return rx

	def union (self, x, y):
		'''Running time: O(log n)'''
		rx = self.find(x)
		ry = self.find(y)
		if rx == ry:
			return False
		elif self._rank[rx] < self._rank[ry]:
			self._parent[rx] = ry
		elif self._rank[ry] < self._rank[rx]:
			self._parent[ry] = rx
		else:
			self._parent[rx] = ry
			self._rank[ry] += 1
		return True

class Graph:
	'''A graph has a set of vertices and a set of edges, with each
	edge being an ordered pair of vertices. '''

	def __init__ (self):
		self._alist = {}
		self._coord = {}

	def add_vertex (self, vertex):
		''' Adds 'vertex' to the graph
		Preconditions: None
		Postconditions: self.is_vertex(vertex) -> True
		'''
		if vertex not in self._alist:
			self._alist[vertex] = set()

	def add_edge (self, source, destination):
		''' Adds the edge (source, destination)
		Preconditions: None
		Postconditions:
		self.is_vertex(source) -> True,
		self.is_vertex(destination),
		self.is_edge(source, destination) -> True
		'''
		self.add_vertex(source)
		self.add_vertex(destination)
		self._alist[source].add(destination)

	def is_edge (self, source, destination):
		'''Checks whether (source, destination) is an edge
		'''
		return (self.is_vertex(source)
				and destination in self._alist[source])

	def is_vertex (self, vertex):
		'''Checks whether vertex is in the graph.
		'''
		return vertex in self._alist

	def neighbours (self, vertex):
		'''Returns the set of neighbours of vertex. DO NOT MUTATE
		THIS SET.
		Precondition: self.is_vertex(vertex) -> True
		'''
		return self._alist[vertex]

	def vertices (self):
		'''Returns a set-like container of the vertices of this
		graph.'''
		return self._alist.keys()

class WeightedGraph (Graph):
	'''A weighted graph stores some extra information (usually a
	"weight") for each edge.'''
	
	def add_vertex (self, vertex, latitude, longitude):
		if vertex not in self._alist:
			self._alist[vertex] = {}
			self._coord[vertex] = (latitude,longitude)

	def add_edge (self, source, destination):
		cost_distance.g = self
		self._alist[source][destination] = cost_distance(source,destination)

	def get_weight (self, source, destination):
		'''Returns the weight associated with this edge.
		Precondition: self.is_edge(source, destination) -> True'''
		return self._alist[source][destination]

	def neighbours (self, vertex):
		'''Returns the set of neighbours of vertex.
		Precondition: self.is_vertex(vertex) -> True
		'''
		return self._alist[vertex].keys()

	def neighbours_and_weights (self, vertex):
		return self._alist[vertex].items()
		
	def vertices (self):
		'''Returns a set-like container of the vertices of this
		graph.'''
		return self._alist.keys()
	
	def is_edge (self, source, destination):
		'''Checks whether (source, destination) is an edge
		'''
		return (self.is_vertex(source)
				and destination in self._alist[source])

	def is_vertex (self, vertex):
		'''Checks whether vertex is in the graph.
		'''
		return vertex in self._alist
		
	
def cost_distance(u,v):
		'''
	Takes u and v as vertices in the graph (g)
	uses Euclidean distance calculation and returns distance in 100,000 degree form
	Requires g as the graph before calling.'''
		return sqrt(abs(cost_distance.g._coord[u][0] - cost_distance.g._coord[v][0])**2
		+abs(cost_distance.g._coord[u][1] - cost_distance.g._coord[v][1])**2)


class UndirectedGraph (WeightedGraph):
	'''An undirected graph has edges that are unordered pairs of
	vertices; in other words, an edge from A to B is the same as one
	from B to A.'''

	def add_edge (self, a, b):
		'''We implement this as a directed graph where every edge has its
		opposite also added to the graph'''
		super().add_edge (a, b)
		super().add_edge (b, a)


def latlon_to_integer(coord):
	return int(float(coord)*100000)

def build_graph():
	g = UndirectedGraph()
	with open("edmonton-roads-2.0.1.txt") as file: # Assumes filename is edmonton-roads-2.0.1.txt
		for line in file: # Variable 'line' loops over each line in the file
			line = line.strip().split(',') # Remove trailing newline character and splits line into list
			if line[0] == 'V':
				g.add_vertex(int(line[1]),latlon_to_integer(line[2]),latlon_to_integer(line[3]))
			if line[0] == 'E':
				g.add_edge(int(line[1]),int(line[2]))
	return g
	
def kruskal (graph, cost):

	edges = [(u, v) for u in graph.vertices() for v in graph.neighbours(u)]

	cost.g = graph
	edges.sort(key=lambda edge: cost(edge[0], edge[1]))

	tree = []
	uf = UnionFind(graph.vertices())

	for (u, v) in edges:
		if uf.union(u, v):
			tree.append((u, v))

	return tree

def least_cost_path(graph, start, dest, cost):
	#Special case: start is dest
	if start==dest:
		print("0.5")
		print(type([].append(start)))
		return [].append(start)
	tree = kruskal(graph,cost)
	reached = [].append(start)
	reached_l = [].append(dest)
	new_graph = tree_to_graph(tree)
	todo = [].append(new_graph.neighbours(start).pop())
	print("1")
	while todo:#first run where it looks for the representative vertex from start
		end = todo.pop()
		if end in new_graph.neighbours(end):
			#If end is pointing to itself, then representative is found and break
			print("1.5")
			break
		#Else add neighbours of end into todo and loop
		end = new_graph.neighbours(end).pop()
		reached.append(end)
		if end == dest:
			#If end is destination, return reached
			print("2")
			return reached
		todo.append(end)
	
	todo = [].append(new_graph.neighbours(dest))
	while todo: #second run where it looks for the representative vertex from dest
		end = todo.pop()
		if end in new_graph.neighbours(end):
			#If end is pointing to itself, then representative is found and break
			print("2.5")
			break
		#Else add neighbours of end into todo and loop
		end = new_graph.neighbours(end).pop()
		reached_l.append(end)
		if end == start:
			#If end is start, return reached
			print("3")
			return reached_l.reverse()
		todo.append(end)

	#Check if representative of start and dest are same, otherwise return empty list
	if reached[-1] != reached[0]:
		print("4")
		return list()
	
	#add start and dest lists together then return final list
	reached_l = reached_l.reverse().pop(0)
	reached = reached + reached_l
	
	print("5")
	print(reached)
	print(type(reached))
	return reached

def tree_to_graph(tree):
	sorted_graph = Graph()
	for vert in tree:
		#~ print(vert[0])
		sorted_graph.add_edge(vert[0], vert[1])
	
	return sorted_graph

def find_cost (edges, cost):
	return sum([cost(u, v) for (u, v) in edges])


#~ def dijkstra(g, v, cost):
	#~ '''
	#~ Find and return the search tree
	#~ obtained by performing Dijkstra's search
	#~ on a graph with edge costs.
	#~ 
	#~ Assumption: g.is_vertex(v) -> True
	#~ cost(u,w) returns the cost of an edge
	#~ (u,w) in the graph.
#~ 
	#~ WARNING: We did not have time to test this code
	#~ in class.
#~ 
	#~ Running time: O(m^2)
	#~ '''
	#~ 
	#~ reached = {}
#~ 
	#~ # just a bandage for now, we will use
	#~ # a better data structure later to get
	#~ # better running time
	#~ runners = {(0, v, v)}
#~ 
	#~ # num iterations <= num edges + 1
	#~ # i.e. O(m)
	#~ while runners:
		#~ # O(m) time per extraction
		#~ # WARNING: if there are multiple "runners"
		#~ # with the same minimum time, then this
		#~ # will compare the second elements of the tuple.
		#~ # Our final, more efficient implementation
		#~ # will avoid this.
		#~ (time, goal, start) = min(runners)
		#~ runners.remove((time, goal, start))
#~ 
		#~ if goal in reached:
			#~ continue
#~ 
		#~ reached[goal] = (start, time)
#~ 
		#~ # O(1) time per insertion, at most m insertions
		#~ # throughout the entire algorithm
		#~ for succ in g.neighbours(goal):
			#~ set.add((time + cost(goal, succ), succ, goal))
#~ 
	#~ return reached
   
	
#~ if __name__ == '__main__':
	#~ # Graph from worksheet
	#~ wg = WeightedGraph()
	#~ edges = [(1, 2, 2), (1, 3, 3), (2, 3, 3),
			 #~ (2, 4, 3), (2, 5, 1), (3, 6, 2),
			 #~ (4, 5, 4), (5, 6, 4), (4, 7, 2), (5, 8, 3), (6, 8, 1), (6, 9, 5),
			 #~ (4, 10, 1), (7, 10, 2), (8, 9, 3), (8, 10, 4), (10, 9, 2)]
	#~ for (u, v, weight) in edges:
		#~ wg.add_edge(u, v)
#~ 
	#~ mst = kruskal(wg, wg.get_weight)
	#~ assert find_cost(mst, wg.get_weight) == 17

def svr(g):
	State = Enum('State', 'R N AN W A E ERR')
	state = State.R

	while state != State.E and state != State.ERR:
		if state == State.R:                                          		 #wait for cli request
			print(state)
			l = input().split()                                         	#get request
			if l[0] == "R":                                         	#if line actualy request
				inp = [int(l[i]) for i in range(1,5)]                   	#convert input to int
				latStart, lonStart, latDest, lonDest = inp[0:4]        	#get data from request
				state = State.N                                         	#next step / first acknowledge
				
		elif state == State.N:                                                #find path
			print(state)
			start, dest = (None, None)
			for vOfC in g._coord.keys():
				if start != None:
					if g._coord[vOfC] == (latStart, lonStart):
						start = g.coord[vOfC]
				elif dest != None:
					if g._coord[vOfC] == (latDest, lonDest):
						dest = g.coord[vOfC]
				else:
					break
			
			cost_distance.g = g
			path = least_cost_path(g, start, dest, cost_distance)                  #get path
			print("N", len(path))                                       #send cli path len
			if len(path) == 0:                                          #if no path
				state = State.R                                         #   wait for new 'R'
			else:                                                       #else
				state = State.AN                                        #   next step
				
		elif state == State.AN:                                               #first acknowledge
			print(state)
			t = input.split()                                           #get N data
			if t[0] == "N" and int(t[1]) != 0:                          #if data is actually N and not 0
				print("A")                                              #   send first acknowledgement
				state = State.W                                         #   next step
				
		elif state == State.W:                                                #send next cords in path
			print(state)
			if len(path) == 0:                                          #if path at dest
				state = State.E                                         #   end step
			elif len(path) != 0:
				point = path.pop(0)                                  #else
				print("W", g._coord[point][0], g._coord[point][1], sep = ' ')                         #   send cords
				state = State.A                                         #   next acknowledge
			
		elif state == State.A:                                                #acknowledge
			print(state)
			inp = input()                                               #get W data
			if inp == "W":                                              #if actualy W data
				print("A")                                              #   send acknowledge
				state = State.W                                         #   next W
			elif inp == "E":                                             #elif E
				state = State.E                                         #   end step
				
		elif state == State.E:                                                #path done
			print("Finished giving path or 'R'")                        #path done
			print(state)                                                #path done
			
		else:                                                           #ERROR
			state = State.ERR                                           #ERROR
			print("There was an error")                                 #ERROR
			print(state)                                                #ERROR
			
	print("Done")

#~ g = build_graph()
#~ svr(g)
