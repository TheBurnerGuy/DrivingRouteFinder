from unionfind import UnionFind

class UndirectedGraph (WeightedGraph):
    '''An undirected graph has edges that are unordered pairs of
    vertices; in other words, an edge from A to B is the same as one
    from B to A.'''

    def add_edge (self, a, b):
        '''We implement this as a directed graph where every edge has its
        opposite also added to the graph'''
        super().add_edge (a, b)
        super().add_edge (b, a)

class WeightedGraph ():
    '''A weighted graph stores some extra information (usually a
    "weight") for each edge.'''
	
	def __init__ (self):
        self._alist = {}
        self._coord = {}
	
    def add_vertex (self, vertex, latitude, longitude):
        if vertex not in self._alist:
            self._alist[vertex] = {}
            self._coord[vertex] = (latitude,longitude)

    def add_edge (self, source, destination):
        self._alist[source][destination] = self.calculate_weight(source,destination)

	def calculate_weight(self, source, destination):
		'''uses Euclidean distance calculation
		'''
		return sqrt(abs(self._coord[source][0] - self._coord[destination][0])**2
		+abs(self._coord[source][1] - self_coord[destination][1])**2)

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

def build_graph():
	g = UndirectedGraph()
	with open("edmonton-roads-2.0.1.txt") as file: # Assumes filename is edmonton-roads-2.0.1.txt
		for line in file: # Variable 'line' loops over each line in the file
			line = line.strip().split(',') # Remove trailing newline character and splits line into list
			if line[0] == 'V':
				g.add_vertex(line[1],line[2],line[3])
			if line[0] == 'E':
				g.add_edge(line[1],line[2])
				
def kruskal (graph, cost):

    edges = [(u, v) for u in graph.vertices() for v in graph.neighbours(u)]
    # Equivalent to following code:
    # for u in graph.vertices():
    #     for v in graph.neighbours(u):
    #         edges.append((u, v))

    # def sort_key (edge):
    #     return cost(edge[0], edge[1])

    # sort_key = lambda edge: cost(edge[0], edge[1])

    edges.sort(key=lambda edge: cost(edge[0], edge[1]))

    tree = []
    uf = UnionFind(graph.vertices())

    for (u, v) in edges:
        if uf.union(u, v):
            tree.append((u, v))

    return tree


def find_cost (edges, cost):
    return sum([cost(u, v) for (u, v) in edges])

if __name__ == '__main__':
    # Graph from worksheet
    wg = WeightedGraph()
    edges = [(1, 2, 2), (1, 3, 3), (2, 3, 3),
             (2, 4, 3), (2, 5, 1), (3, 6, 2),
             (4, 5, 4), (5, 6, 4), (4, 7, 2), (5, 8, 3), (6, 8, 1), (6, 9, 5),
             (4, 10, 1), (7, 10, 2), (8, 9, 3), (8, 10, 4), (10, 9, 2)]
    for (u, v, weight) in edges:
        wg.add_edge(u, v, weight)

    mst = kruskal(wg, wg.get_weight)
    assert find_cost(mst, wg.get_weight) == 17
