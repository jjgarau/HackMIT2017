import numpy as np


def dot_Kmeans(verts, horizs, nodes, max_coords, niter=10, ntimes=10):
    clusters_s = {}

    print('NODEEES v2')
    print(nodes)

    for exec in range(ntimes):
        xs = np.random.uniform(low=0, high=max_coords[1], size=len(horizs))
        ys = np.random.uniform(low=0, high=max_coords[0], size=len(verts))

        print(xs.shape, ys.shape)

        n_clusters = len(horizs) + len(verts)
        clusters = np.array([-1] * len(nodes), dtype=int)  # assign each point to a cluster
        dist_clust = np.array([np.inf] * len(nodes))

        for k in range(niter):

            # maximization
            for cluster in range(len(horizs)):
                d = nodes - np.array([xs[cluster], horizs[cluster]])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if clusters[j] < 0 or d[j] < dist_clust[j]:
                        clusters[j] = cluster
                        dist_clust[j] = d[j]

            for cluster in range(len(verts)):
                d = nodes - np.array([verts[cluster], ys[cluster]])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if clusters[j] < 0 or d[j] < dist_clust[j]:
                        clusters[j] = cluster + len(horizs)
                        dist_clust[j] = d[j]

            dist = np.sum(dist_clust)
            print('TOT DISTANCE!', dist)

            # expectation
            for cluster in range(len(horizs)):
                members = clusters == cluster
                if np.any(members):
                    xs[cluster] = np.mean(nodes[members, 0])
                else:
                    xs[cluster] = np.random.uniform(low=0, high=max_coords[1])
            for cluster in range(len(verts)):
                members = clusters == (cluster + len(horizs))
                if np.any(members):
                    ys[cluster] = np.mean(nodes[members, 1])
                else:
                    ys[cluster] = np.random.uniform(low=0, high=max_coords[0])


        key = tuple(clusters)

        if not key in clusters_s:
            clusters_s[key] = (1, xs, ys, dist)
        elif dist < clusters_s[key][3]:
            clusters_s[key] = (clusters_s[key][0] + 1, xs, ys, dist)
        else:
            clusters_s[key] = (clusters_s[key][0] + 1, clusters_s[key][1], clusters_s[key][2], clusters_s[key][3])

    clusters, n_max, xs, ys, dist = [], 0, [], [], np.inf
    for cluster, (n, x, y, d) in clusters_s.items():
        if n > n_max:
            n_max = n
            clusters = cluster
            xs = x
            ys = y
            if d > dist:
                print('it is worse in distance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            dist = d

    return clusters, xs, ys, dist