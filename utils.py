
def cartesian(listx, listy):
    ret = []
    for x in listx:
        for y in listy:
            ret.append((x,y))
    return ret

def find_edges(coords):
    edges = []
    lgth = len(coords)
    for i in range(lgth):
        for j in range(i+1, lgth):
            ci = coords[i]
            cj = coords[j]
            if (ci[0] == cj[0]) ^ (ci[1] == cj[1]):
                edges.append((i,j))
    return edges

print(find_edges([(1,1),(4,1),(1,5),(5,5)]))