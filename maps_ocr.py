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
from dot_Kmeans import *

base_uri = r'/Users/miguelperezsanchis/Downloads/hackMIT/png_maps'
img_path = os.path.join(base_uri, '1_0.png')
node_path = os.path.join(base_uri, '1_0_nodes.pickle')

img = Image.open(img_path).convert('L')

pix = np.array(img) * 1. / 255.

# pix = f.canny(pix)

nodeset = NodeSet(img_path, filepath=node_path)

pix = nodeset.map_
print(nodeset.nodes.shape)

for _ in range(20):
    pix = morph.binary_dilation(pix)

print(pix.shape)

h, theta, d = hough_line(pix, theta=np.linspace(start=-np.pi/2., stop=np.pi/2, num=13))

print(h.shape, theta.shape, d.shape)

h, angles, dists = hough_line_peaks(h, theta, d, min_distance=20, num_peaks=12, threshold=0.5*h.max())

clust_v, clust_h, xs, ys, cost = find_best_Kmeans(angles, dists, pix.shape)

print('KMEANS')
print(clust_v)
print(clust_h)
print(xs)
print(ys)
print(cost)

corners = [(y, x) for y in clust_h for x in clust_v]

clust_dv, clust_dh, dxs, dys, dcost_v, dcost_h = dot_Kmeans(xs, ys, nodeset.nodes, pix.shape)

print('KMEANS_DOT')
print(clust_dv)
print(clust_dh)
print(dxs)
print(dys)
print(dcost_v)
print(dcost_h)

dots_in_lines = list(zip(xs, dys)) + list(zip(dxs, ys))




fig, ax = plt.subplots()

for _, angle, dist in zip(h, angles, dists):
    y0 = (dist - 0 * np.cos(angle)) / (np.sin(angle) + 0.0001)
    y1 = (dist - pix.shape[1] * np.cos(angle)) / (np.sin(angle) + 0.001)
    ax.plot((0, pix.shape[1]), (y0, y1), '-r')

for x in xs:
    ax.plot((x, x), (0, pix.shape[0]), '-b')
for y in ys:
    ax.plot((0, pix.shape[1]), (y,y), '-b')

nods = np.array(nodeset.nodes)
ax.plot(nods[:,1], nods[:,0], 'ro')

for (x, dy) in zip(xs, dys):
    ax.plot(x,dy, 'g^')
for (dx, y) in zip(dxs, ys):
    ax.plot(dx,y, 'g^')

ax.set_xlim((0, pix.shape[1]))
ax.set_ylim((pix.shape[0], 0))
ax.set_axis_off()
ax.set_title('Detected lines')
ax.imshow(pix, cmap=plt.cm.gray)
ax.set_title('Input image')

plt.tight_layout()
mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

