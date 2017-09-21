import json

def save_json(obj, fpath):
    with open(fpath, 'w') as f:
        json.dump(obj)

def load_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def approx_eq(a, b, tol):
    return abs(a - b) < tol

def find_edges(coords):
    edges = []
    lgth = len(coords)
    for i in range(lgth):
        for j in range(i+1, lgth):
            ci = coords[i]
            cj = coords[j]
            if approx_eq(ci[0], cj[0], 1e-3) ^ approx_eq(ci[1], cj[1], 1e-3):
                print(ci, cj)
                edges.append((i,j))
    return edges

# print(find_edges([(1,1),(4,1),(1,5),(5,5)]))


def build_graph_from_edges(edges, n_nodes):
    g = [[] for _ in range(n_nodes)]
    for edge in edges:
        g[edge[0]].append(edge[1])
    return g


def build_whole_graph(nodes_coords, node2cluster, g_ini):
    shift_id = len(g_ini)
    n_nodes = len(nodes_coords)
    g = g_ini + [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        print(node2cluster[i], '->', i+shift_id)
        g[node2cluster[i]].append(i+shift_id)
    return g


# def get_graph_structure(nodes_coords, g):
#     G = [[] for _ in range(len(g))]
#     for i, (x, y) in enumerate(nodes_coords):
#         for j in g[i]:
