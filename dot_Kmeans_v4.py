import numpy as np


def dot_Kmeans(verts, horizs, nodes, max_coords, niter=10, ntimes=10):

    final_clust, final_xs, final_ys, final_dist, final_exec = [], [], [], np.inf, -1

    print('NODEEES v4')
    print(nodes)
    np.set_printoptions(precision=2, linewidth=300, threshold=np.inf)

    for exec in range(ntimes):
        xs = np.random.uniform(low=0, high=max_coords[1], size=len(horizs))
        ys = np.random.uniform(low=0, high=max_coords[0], size=len(verts))

        print(xs.shape, ys.shape)
        print(len(nodes))

        n_clusters = len(horizs) + len(verts)
        node2cluster = np.array([-1] * len(nodes), dtype=int)  # assign each point to a cluster

        for k in range(niter):

            print('k', k)
            ds = np.zeros((n_clusters, len(nodes)))
            cluster_coords = [None] * n_clusters
            # maximization
            for cluster in range(n_clusters):
                for j in range(len(nodes)):
                    cluster_coords[cluster] = [xs[cluster], horizs[cluster]] if cluster < len(horizs) else [verts[cluster - len(horizs)], ys[cluster-len(horizs)]]
                    d = nodes[j, :] - np.array(cluster_coords[cluster])
                    ds[cluster, j] = np.sqrt(np.sum(d*d))

            print(np.array(cluster_coords))
            print(np.vstack((nodes[:,0], nodes[:,1], ds, np.argmin(ds, axis=0))).T)
            node2cluster = np.argmin(ds, axis=0)
            dist_clust = np.min(ds, axis=0)
            dist = np.sum(dist_clust)

            # expectation
            for cluster in range(n_clusters):
                tot = 0.0
                cont = 0
                for j in range(len(nodes)):
                    if node2cluster[j] == cluster:
                        tot += nodes[j, 0] if cluster < len(horizs) else nodes[j, 1]
                        cont += 1
                if cluster < len(horizs):
                    if tot:
                        xs[cluster] = tot / cont
                    else:
                        xs[cluster] = np.random.uniform(low=0, high=max_coords[1])
                else:
                    if tot:
                        ys[cluster - len(horizs)] = tot / cont
                    else:
                        ys[cluster - len(horizs)] = np.random.uniform(low=0, high=max_coords[0])

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