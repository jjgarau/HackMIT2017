#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import dilation, opening
from skimage.measure import regionprops
from sklearn.cluster import KMeans


class NodeSet():

    def __init__(self, file, nodes=True, n_ext=None):
        self.map_ = self.get_map(file)
        if nodes:
            self.p_nodes = self.get_pnodes()
            self.nodes = self.get_nodes()
        else:
            self.p_nodes = None
            self.nodes = n_ext

    def get_map(self, file):
        img = Image.open(file).convert('L')
        pix = np.array(img) * 1. / 255.
        pix = (pix < 0.7).astype(int)
        pix = pix[200:-600, 200:-200]
        bb = regionprops(pix)[0].bbox
        pix = pix[bb[0]:bb[2], bb[1]:bb[3]]
        print("Map created!")
        return pix

    def get_pnodes(self):
        corners = np.array(corner_peaks(corner_harris(self.map_), min_distance=1))
        print("Corners located!")
        corner_map = np.zeros(self.map_.shape)
        for corner in corners:
            corner_map[corner[0], corner[1]] = 1
        selem_mat = np.ones((12, 12))
        for _ in range(5):
            corner_map = dilation(corner_map)
        for _ in range(1):
            corner_map = opening(corner_map, selem=selem_mat)
        for _ in range(8):
            corner_map = dilation(corner_map)
        print("Pseudo-nodes located!")
        p_nodes = []
        for i in range(len(corner_map)):
            for j in range(len(corner_map[i])):
                if corner_map[i, j] > 0.5:
                    p_nodes.append([i, j])
        return p_nodes

    def get_nodes(self):
        inertias = []
        inertias_d = []
        n_c_min = 25
        n_c_max = 32
        for i in range(n_c_min, n_c_max):
            print('Clustering...' + str(i-n_c_min+1) + '/' + str(n_c_max-n_c_min))
            kmeans = KMeans(n_clusters=i).fit(np.array(self.p_nodes))
            inertias.append(kmeans.inertia_)
            if i != n_c_min:
                inertias_d.append((kmeans.inertia_ - inertias[-2])/inertias[-1])
        n_c_o = np.argmin(inertias_d) + n_c_min + 1
        print("Optimal Cluster Quantity: " + str(n_c_o))
        kmeans = KMeans(n_clusters=n_c_o).fit(np.array(self.p_nodes))
        return kmeans.cluster_centers_
