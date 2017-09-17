import numpy as np

def read_nodes_coords(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
    return np.array([list(reversed([float(x) for x in line.split()])) for line in lines[4:]]) # format (x, y)


def dot_Kmeans(verts, horizs, nodes, max_coords, niter, ntimes):
    clusters_hs = {}
    clusters_vs = {}

    for exec in range(ntimes):
        xs = np.random.uniform(low=0, high=max_coords[1], size=len(horizs))
        ys = np.random.uniform(low=0, high=max_coords[0], size=len(verts))

        print(xs.shape, ys.shape)

        dist_v = 0.0
        dist_h = 0.0

        clusters_h = np.array([-1] * len(nodes), dtype=int)  # assign each point to a cluster
        dist_clust_h = np.array([np.inf] * len(nodes))
        clusters_v = np.array([-1] * len(nodes), dtype=int)
        dist_clust_v = np.array([np.inf] * len(nodes))

        for k in range(niter):
            # maximization
            for i, x in enumerate(xs):
                d = nodes - np.array([x, verts[i]])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if clusters_h[j] < 0 or d[j] < dist_clust_h[j]:
                        print(d[j], dist_clust_h[j])
                        clusters_h[j] = i
                        dist_clust_h[j] = d[j]
                dist_h += d

            for i, y in enumerate(ys):
                d = nodes - np.array([horizs[i], y])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if clusters_v[j] < 0 or d[j] < dist_clust_v[j]:
                        print(d[j], dist_clust_v[j])
                        clusters_v[j] = i
                        dist_clust_v[j] = d[j]
                dist_v += d

            # expectation
            for clust in range(len(clusters_h)):
                members = clusters_h == clust
                if np.any(members):
                    print(members, nodes)
                    xs[clust] = np.mean(nodes[members, 0])
            for clust in range(len(clusters_v)):
                members = clusters_v == clust
                if np.any(members):
                    print(members, nodes)
                    ys[clust] = np.mean(nodes[members,1])

        clusters_h, n_max, xs, dist_h = [], 0, [], 0.0
        for cluster, (n, x, d) in clusters_hs.items():
            if n > n_max:
                n_max = n
                clusters_h = cluster
                xs = x
                dist_h = d

        clusters_v, n_max, ys, dist_v = [], 0, [], 0.0
        for cluster, (n, y, d) in clusters_vs.items():
            if n > n_max:
                n_max = n
                clusters_v = cluster
                ys = y
                dist_v = d



        return clusters_v, clusters_h, xs, ys, dist_v, dist_h