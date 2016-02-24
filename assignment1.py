from enum import Enum
from math import sqrt
import argparse
import textserial
import time

class MinHeap:
    def __init__(self):
        self._array = []

    def add(self,key,value):
        self._array.append( (key,value) )
        self.fix_heap_up( len(self._array)-1 )
        
    def pop_min(self):
        if not self._array:
            raise RuntimeError("Attempt to call pop_min on empty heap")
        r = self._array[0] #??
        l = self._array[-1]
        del self._array[-1]
        if self._array:
            self._array[0] = l
            self.fix_heap_down(0)
        return r
        
    def fix_heap_up(self,i):
        if self.isroot(i):
            return
        p = self.parent(i)
        if self._array[i][0]<self._array[p][0]:
            self.swap(i,p)
            self.fix_heap_up(p)
            
    def swap(self,i,j):
        self._array[i],self._array[j] = \
            self._array[j],self._array[i]
            
    def isroot(self,i):
        return i==0
        
    def isleaf(self,i):
        return self.lchild(i)>=len(self._array)
        
    def lchild(self,i):
        return 2*i+1
        
    def rchild(self,i):
        return 2*i+2
        
    def parent(self,i):
        return (i-1)//2
        
    def min_child_index(self,i):
        l = self.lchild(i)
        r = self.rchild(i)
        retval = l
        if r<len(self._array) and self._array[r][0]<self._array[l][0]:
            retval = r
        return retval
        
    def isempty(self):
        return len(self._array)==0
    
    def fix_heap_down(self,i):
        if self.isleaf(i):
            return
            
        j = self.min_child_index(i)
        if self._array[i][0]>self._array[j][0]:
            self.swap(i,j)
            self.fix_heap_down(j)

class Graph:
	'''A graph has a set of vertices and a set of edges, with each
	edge being an ordered pair of vertices. '''

	def __init__ (self):
		self._alist = {}
		self._coord = {} # holds lat and lon values

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
			self._coord[vertex] = (latitude,longitude) #holds lat and lon values

	def add_edge (self, source, destination):
		'''Adds an edge to graph and uses cost_distance() to calculate edge weight'''
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

def find_closest_vertex(graph, lat, lon):
	'''Takes closes vertex to selected lat and lon location based on graph'''
	closest = -1
	closestDistance = 0
	for vertex in graph._alist.keys():
		if closest==-1:
			closest = vertex
			closestDistance = sqrt(abs(graph._coord[vertex][0] - lat)**2+abs(graph._coord[vertex][1] - lon)**2)
		else:
			distance = sqrt(abs(graph._coord[vertex][0] - lat)**2+abs(graph._coord[vertex][1] - lon)**2)
			if distance < closestDistance:
				closest = vertex
				closestDistance = distance
	print(closest)#Debug code to test closest vertex
	return closest

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
	'''Takes edmonton-roads-2.0.1.txt and uses it as instructions to create an undirectedgraph.'''
	g = UndirectedGraph()
	with open("edmonton-roads-2.0.1.txt") as file: # Assumes filename is edmonton-roads-2.0.1.txt
		for line in file: # Variable 'line' loops over each line in the file
			line = line.strip().split(',') # Remove trailing newline character and splits line into list
			if line[0] == 'V':
				g.add_vertex(int(line[1]),latlon_to_integer(line[2]),latlon_to_integer(line[3]))
			if line[0] == 'E':
				g.add_edge(int(line[1]),int(line[2]))
	return g
	
def least_cost_path(graph, start, dest, cost):
	'''
	Find and return the least cost path between start and dest using
	Djlkstra's algorithm, weighted edges, and the minheap property.
	
	Assumption: g.is_vertex(v) -> True
	cost(u,w) returns the cost of an edge
	(u,w) in the graph.
	
	Running time: O(log(m))
	'''
	
	if start == dest:
		return []
	
	#Creates reached and runners data structure
	reached = {}
	runners = MinHeap()
	#Initializes runners using start vertices
	runners.add((start, start), 0)
	cost.g = graph
	
	#Keep running until all runners have dispersed or goal is reached
	while runners:
		#Pop lowest value in runners
		((start, goal), time) = runners.pop_min()
		
		#If already found goal, loop until no more runners.
		if goal in reached:
			continue
		
		#Adds path into reached if not in reached already
		reached[goal] = start
		#If found dest, break
		if goal == dest:
			break
		
		#Adds all neighbours of goal as a runner
		for succ in graph.neighbours(goal):
			runners.add((goal, succ), (time + cost(goal, succ)))
	else:
		#If dest not found, return empty list
		return []
	#Create a returnable list
	path = []
	vertex = dest
	while True:
		#Create list based off of compiled dictionary from dest to start
		path.insert(0,vertex)
		if vertex == start:
			break
		vertex = reached[vertex]
	#Return path
	return path

def parse_args():
    """
    Parses arguments for this program.
    
    Returns:
    An object with the following attributes:
     serialport (str): what is after -s or --serial on the command line
    """
    # try to automatically find the port
    port = textserial.get_port()
    if port==None:
        port = "0"

    parser = argparse.ArgumentParser(
          description='Serial port communication testing program.'
        , epilog = 'If the port is 0, stdin/stdout are used.\n'
        )
    parser.add_argument('-s', '--serial',
                        help='path to serial port '
                             '(default value: "%s")' % port,
                        dest='serialport',
                        default=port)

    return parser.parse_args()

def srv(g, serial_in, serial_out):
	'''Args = Graph(), port in, port out: 
	Simple server side statemachine'''
	
	State = Enum('State', 'R N AN W A E ERR')
	state = State.R
	ser.setTimeout(1)
	line = "a b"

	while state != State.ERR:
		if state == State.R:
			'''Wait for client request'''
			#print(state) #DEBUG: print current state
			line = "a"
			while line[0] != 'R':
				'''Check if input is not request, then clear buffer'''
				line = serial_in.readline()
				line = line.strip('\r\n')
				print("Buffer: <%s>" % line) #DEBUG: print line entered when not request
				if len(line) == 0:
					line = "a"
			'''Now that input is actualy the request, take coordinates from request line'''
			line = line.split();
			inp = [int(line[i]) for i in range(1,5)]
			latStart, lonStart, latDest, lonDest = inp[0:4]
			closestStartVertex = find_closest_vertex(g, latStart, lonStart)
			closestEndVertex = find_closest_vertex(g, latDest, lonDest)
			latLonStart = g._coord[closestStartVertex]
			latStart = int(latLonStart[0])
			lonStart = int(latLonStart[1])
			latLonDest = g._coord[closestEndVertex]
			latDest = int(latLonDest[0])
			lonDest = int(latLonDest[1])
			print(latStart, lonStart, latDest, lonDest) #DEBUG: print coordinates entered
			state = State.N
				
		elif state == State.N:
			'''Find the path and send client the number of vertices'''
			#print(state) #DEBUG: print current state
			start = closestStartVertex
			dest = closestEndVertex
			cost_distance.g = g
			path = least_cost_path(g, start, dest, cost_distance)
			'''Lookup the path'''
			print("N " + str(len(path)), file = serial_out)
			print("N " + str(len(path))) #DEBUG: print path length
			if len(path) == 0:
				'''if the path length returns 0 then there is no path, wait for new request'''
				state = State.R
			else:
				'''else, wait for client to acknowledge data received'''
				state = State.AN
				
		elif state == State.AN:
			'''Wait for arduino to acknowledge it got N ... (TIMEOUT = 1)'''
			#print(state) #DEBUG: print current state
			timeout = 0
			while line[0] != 'A' and timeout < 1:
				'''Check if input is not request, then clear buffer'''
				timeout += 1
				time.sleep(1)
				line = serial_in.readline()
				line = line.strip('\r\n')
				if len(line) == 0:
					line = "d"
				print("BufferAN: <%s>" % line) #DEBUG: print line entered when not request
				
			if line[0] == 'A':
				'''If what is read is acknowledgement, countinue'''
				state = State.W
			else:
				'''else reset statemachine'''
				state = State.R
				
		elif state == State.W:
			#print(state) #DEBUG: print current state
			#~ if len(path) == 0:
				#~ '''if path at destination, finish'''
			#~ state = State.E
			#~ elif len(path) != 0:
				#~ '''else, pop off next vertex,
						#~ lookup coordinates for said vertex,
						#~ send to client coordinates'''
			point = path.pop(0)
			print("W", g._coord[point][0], g._coord[point][1], sep = ' ', file = serial_out)
			print("W", g._coord[point][0], g._coord[point][1], sep = ' ') 
			#DEBUG: print coordinates of lookedup vertex
			state = State.A
			if len(path) == 0:
				state = State.E
			
		elif state == State.A:
			#print(state) #DEBUG: print current state
			timeout = 0
			line = serial_in.readline().strip('\r\n')
			while line[0] != 'A' and timeout < 1:
				'''Check if input is not request, then clear buffer'''
				timeout += 1
				time.sleep(1)
				line = serial_in.readline()
				line = line.strip('\r\n')
				if len(line) == 0:
					line = "e"
				print("BufferA: <%s>" % line) #DEBUG: print line entered when not request
				
			if line[0] == 'A':
				'''If what is read is acknowledgement, countinue'''
				state = State.W
			else:
				'''else reset statemachine'''
				state = State.R
				
		elif state == State.E:
			'''Debug state, print that done, pritn that returning to state.R and return to 'R' '''
			#print(state) #DEBUG: print current state
			#print("E", file = serial_out)
			print("Finished giving path or 'R'")
			print("Waiting for new request back in state 'R'")
			state = State.R
			
		else:
			'''Code should never get here unless error in code'''
			state = State.ERR
			#print(state) #DEBUG: print current state
			print("There was an error")

g = build_graph()
args = parse_args()
           
if args.serialport!="0":
    print("Opening serial port: %s" % args.serialport)
    baudrate = 9600 # [bit/seconds] 115200 also works
    with textserial.TextSerial(args.serialport,baudrate,newline=None) as ser:
        srv(g,ser,ser)
