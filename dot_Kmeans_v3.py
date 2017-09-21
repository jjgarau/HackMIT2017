import numpy as np


def dot_Kmeans(verts, horizs, nodes, max_coords, niter=10, ntimes=10):

    final_clust, final_xs, final_ys, final_dist, final_exec = [], [], [], np.inf, -1

    print('NODEEES v3')
    print(nodes)

    for exec in range(ntimes):
        xs = np.random.uniform(low=0, high=max_coords[1], size=len(horizs))
        ys = np.random.uniform(low=0, high=max_coords[0], size=len(verts))

        print(xs.shape, ys.shape)
        print(len(nodes))

        n_clusters = len(horizs) + len(verts)
        node2cluster = np.array([-1] * len(nodes), dtype=int)  # assign each point to a cluster
        dist_clust = np.array([np.inf] * len(nodes))

        for k in range(niter):

            print('k', k)

            # maximization
            for cluster in range(len(horizs)):
                d = nodes - np.array([xs[cluster], horizs[cluster]])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if d[j] < dist_clust[j]:
                        node2cluster[j] = cluster
                        dist_clust[j] = d[j]

            for cluster in range(len(verts)):
                d = nodes - np.array([verts[cluster], ys[cluster]])
                d = np.sum(d*d, axis=1)
                for j in range(len(d)):
                    if d[j] < dist_clust[j]:
                        node2cluster[j] = cluster + len(horizs)
                        dist_clust[j] = d[j]

            dist = np.sum(dist_clust)
            print('TOT DISTANCE!', dist)

            # expectation
            for cluster in range(len(horizs)):
                members = node2cluster == cluster
                if np.any(members):
                    xs[cluster] = np.mean(nodes[members, 0])
                else:
                    xs[cluster] = np.random.uniform(low=0, high=max_coords[1])
            for cluster in range(len(verts)):
                members = node2cluster == (cluster + len(horizs))
                if np.any(members):
                    ys[cluster] = np.mean(nodes[members, 1])
                else:
                    ys[cluster] = np.random.uniform(low=0, high=max_coords[0])

            # recalculate distances of each point to its cluster
            cluster_coords = [[xs[cluster], horizs[cluster]] if cluster < len(horizs) else [verts[cluster - len(horizs)], ys[cluster-len(horizs)]] for cluster in node2cluster]
            dist_clust = nodes - np.array(cluster_coords)
            dist_clust = np.sum(dist_clust*dist_clust, axis=1)

        if dist < final_dist:
            final_clust = node2cluster
            final_xs = xs
            final_ys = ys
            final_dist = dist
            final_exec = exec

    print('FINAL exec', final_exec)

    cluster_coords = []
    for cluster in range(n_clusters):
        cluster_coords.append([final_xs[cluster], horizs[cluster]] if cluster < len(horizs) else [
            verts[cluster - len(horizs)], final_ys[cluster - len(horizs)]])

    return final_clust, cluster_coords, final_dist