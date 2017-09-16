#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx


class Graph():

    def __init__(self, structure):
        self.structure = structure
        self.G = nx.freeze(self.create_graph(structure))
        self.nn = nx.number_of_nodes(self.G)
        self.ne = nx.number_of_edges(self.G)

    def create_graph(self, structure):
        G = nx.Graph()
        G.add_nodes_from(range(len(structure)))
        for node, edges in enumerate(structure):
            for edge in edges:
                G.add_edge(node, edge[0], distance=edge[1])
        return G

    def get_dijkstra_path(self, source, target, dist=True):
        path = nx.dijkstra_path(self.G, source, target)
        if dist:
            distance = self.get_path_distance(path)
            return path, distance
        return path

    def get_astar_path(self, source, target, dist=True):
        path = nx.astar_path(self.G, source, target)
        if dist:
            distance = self.get_path_distance(path)
            return path, distance
        return path

    def get_path_distance(self, path):
        distance = 0
        last = path[0]
        for node in path[1:]:
            if last < node:
                small, big = last, node
            else:
                small, big = node, last
            for edge in self.structure[small]:
                if edge[0] == big:
                    distance += edge[1]
                    break
            last = node
        return distance

    def get_central_node(self):
        return nx.center(self.G)

if __name__ == "__main__":
    nodeset = [
            [(3,1)],
             [(2,1)],
              [(3,1),(5,1.4)],
               [(4,1),(5,1)],[],
                [(6,1),(7,1.4)],
                 [(8,1)]]
    gr = Graph(nodeset)
    print(gr.nn)
    print(gr.ne)
    path, dist = gr.get_dijkstra_path(1, 7)
    print(path)
    print(dist)
    print(gr.get_central_node())
