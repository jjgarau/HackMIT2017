import json
from pprint import pprint
import math

def dist(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)

with open('coords.txt') as data_file:
    data = json.load(data_file)
    nnodes = len(data["coords"])
    structure = [[] for i in range(nnodes)]
    for edge in data["adj"]:
        a = edge[0]
        b = edge[1]
        d = dist( data["coords"][a][0], data["coords"][a][1], data["coords"][b][0],
                  data["coords"][b][1])
        structure[a].append([b, d])

    print(structure)

