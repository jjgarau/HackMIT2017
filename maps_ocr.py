import os
import numpy as np
import mpld3
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature as f
from skimage import morphology as morph
from skimage.transform import hough_line, hough_line_peaks

from nodes import NodeSet

from rect_Kmeans import *
from dot_Kmeans_v3 import *
from utils import *

fname = '1_1'

base_uri = r'/Users/miguelperezsanchis/Downloads/hackMIT'
img_path = os.path.join(base_uri, 'png_maps', fname + '.png')
node_path = os.path.join(base_uri, 'nodes_maps', fname +'_nodes.pickle')
graph_path = os.path.join(base_uri, 'graph_maps', fname + '.json')


nodeset = NodeSet(img_path, nodes=False, filepath=node_path)

nodes_coords = np.flip(nodeset.nodes, axis=1)  # now is (x,y)
pix = nodeset.map_

for _ in range(20):
    pix = morph.binary_dilation(pix)

print(pix.shape)

h, theta, d = hough_line(pix, theta=np.linspace(start=-np.pi/2., stop=np.pi/2, num=13))

print(h.shape, theta.shape, d.shape)

h, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, num_peaks=12, threshold=0.3*h.max())

clust_v, clust_h, xs, ys, cost = find_best_Kmeans(angles, dists, pix.shape)

print('KMEANS')
print(clust_v)
print(clust_h)
print(xs)
print(ys)
print(cost)

node2cluster, cluster_coords, dcost = dot_Kmeans(xs, ys, nodes_coords, pix.shape, niter=5, ntimes=50)

print('KMEANS_DOT')
print(node2cluster)
print(cluster_coords)
print(dcost)

corners = [(x, y) for x in xs for y in ys]
print('CORNERS\n', corners)

dots_in_lines = np.array(cluster_coords + corners)  # coordinates of important nodes

print('DOTS IN LINES\n', len(dots_in_lines))
print(dots_in_lines)

g_important = build_graph_from_edges(find_edges(dots_in_lines), len(dots_in_lines))

print('[')
for l in g_important:
    print('\t', l)
print(']')

g_whole = build_whole_graph(nodes_coords, node2cluster, g_important)

graph_coords = np.concatenate((dots_in_lines, nodes_coords))

graph = {'name': fname, 'coords': graph_coords, 'struc': g_whole}
save_json(graph, graph_path)

fig, ax = plt.subplots()

# plot hough lines result
for _, angle, dist in zip(h, angles, dists):
    y0 = (dist - 0 * np.cos(angle)) / (np.sin(angle) + 0.0001)
    y1 = (dist - pix.shape[1] * np.cos(angle)) / (np.sin(angle) + 0.0001)
    ax.plot((0, pix.shape[1]), (y0, y1), '-r')

# plot rects kmeans result
for x in xs:
    ax.plot((x, x), (0, pix.shape[0]), '-b')
for y in ys:
    ax.plot((0, pix.shape[1]), (y,y), '-b')

# plot nodes found by clustering
ax.plot(nodes_coords[:,0], nodes_coords[:,1], 'ro')

# plot corridors nodes found by dotsKmeans
for (x, y) in cluster_coords:
    ax.plot(x,y, 'g^')

# plot whole gragh
for i, (x, y) in enumerate(graph_coords):
    plotted = []
    for j in g_whole[i]:
        xj, yj = graph_coords[j]
        ax.plot((x, xj), (y, yj), '--y')

ax.set_xlim((0, pix.shape[1]))
ax.set_ylim((pix.shape[0], 0))
ax.set_axis_off()
ax.imshow(pix, cmap=plt.cm.gray)
ax.set_title('Input image')

plt.tight_layout()
mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

