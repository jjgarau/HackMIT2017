import numpy as np


def orient_vert(theta):
    if -5*np.pi/180 < theta < 5*np.pi/180:
        return True
    else:
        return False


def get_orient_xy(thetas, dists):
    print(thetas, dists)
    o_v = np.array([orient_vert(theta) for theta in thetas], dtype=bool)
    print(o_v)
    verts = (dists*np.abs(np.abs(np.cos(thetas))))[o_v] if np.any(o_v) else np.array([])
    horizs = (dists*np.abs(np.sin(np.abs(thetas))))[np.logical_not(o_v)]
    print(verts, horizs)
    return verts, horizs


# rects_td is tuple of thetas, dist
def find_best_Kmeans(thetas, dists, max_coords, niter=10, ntimes=10):
    verts, horizs = get_orient_xy(thetas, dists)
    all_kmeans = []
    costs = []
    improv = []
    if not verts.any():
        for j in range(1, 4):
            clusters_v, clusters_h, xs, ys, dist_v, dist_h = Kmeans(verts, horizs, 0, j, max_coords, niter, ntimes)
            all_kmeans.append((clusters_v, clusters_h, xs, ys))
            costs.append(dist_v + dist_h)
            if j != 1:
                improv.append((costs[-1] - costs[-2]) * 1. / costs[-1])

    elif not horizs.any():
        for i in range(1, 4):
            clusters_v, clusters_h, xs, ys, dist_v, dist_h = Kmeans(verts, horizs, i, 0, max_coords, niter, ntimes)
            all_kmeans.append((clusters_v, clusters_h, xs, ys))
            costs.append(dist_v + dist_h)
            if i != 1:
                improv.append((costs[-1] - costs[-2]) * 1. / costs[-1])
    else:
        for i in range(1,4):
            for j in range(1,4):
                clusters_v, clusters_h, xs, ys, dist_v, dist_h = Kmeans(verts, horizs, i, j, max_coords, niter, ntimes)
                all_kmeans.append((clusters_v, clusters_h, xs, ys))
                costs.append(dist_v + dist_h)
                if i != 0 and j > 1:
                    improv.append((costs[-1] - costs[-2]) * 1. / costs[-1])
    best_cost = np.argmin(costs)
    # return all_kmeans[best_cost] + (costs[best_cost],)
    if best_cost == 0:
        return all_kmeans[0] + (costs[0],)
    else:
        k = np.argmin(improv)
        return all_kmeans[k+1] + (costs[0],)


def Kmeans(vert_rects, horiz_rects, n_vert, n_horiz, max_coords, niter=10, ntimes=10):

    clusters_hs = {}
    clusters_vs = {}

    for exec in range(ntimes):

        xs = np.random.uniform(low=0, high=max_coords[1], size=n_vert)
        ys = np.random.uniform(low=0, high=max_coords[0], size=n_horiz)

        print(xs.shape, ys.shape)

        dist_v = 0.0
        dist_h = 0.0
        clusters_h = np.array([-1] * len(horiz_rects), dtype=int) # assign each point to a cluster
        dist_clust_h = np.array([np.inf] * len(horiz_rects))
        clusters_v = np.array([-1] * len(vert_rects), dtype=int)
        dist_clust_v = np.array([np.inf] * len(vert_rects))

        for k in range(niter):
            # maximization
            for i, x in enumerate(xs):
                d = np.abs(vert_rects - x)
                for j in range(len(d)):
                    if clusters_v[j] < 0 or d[j] < dist_clust_v[j]:
                        print(d[j], dist_clust_v[j])
                        clusters_v[j] = i
                        dist_clust_v[j] = d[j]
            dist_v += np.sum(dist_clust_v * dist_clust_v)

            for i, y in enumerate(ys):
                d = np.abs(horiz_rects - y)
                for j in range(len(d)):
                    if clusters_h[j] < 0 or d[j] < dist_clust_h[j]:
                        print(d[j], dist_clust_h[j])
                        clusters_h[j] = i
                        dist_clust_h[j] = d[j]
            dist_h += np.sum(dist_clust_h * dist_clust_h)

            # expectation
            for clust in range(len(clusters_v)):
                members = clusters_v == clust
                if np.any(members):
                    print(members, vert_rects)
                    xs[clust] = np.mean(vert_rects[members])
            for clust in range(len(clusters_h)):
                members = clusters_h == clust
                if np.any(members):
                    print(members, horiz_rects, horiz_rects[members])
                    ys[clust] = np.mean(horiz_rects[members])


        key_v = tuple(clusters_v)
        key_h = tuple(clusters_h)

        if not key_v in clusters_vs:
            clusters_vs[key_v] = (1, xs, dist_v)
        elif dist_v < clusters_vs[key_v][2]:
            clusters_vs[key_v] = (clusters_vs[key_v][0]+1, xs, dist_v)
        else:
            clusters_vs[key_v] = (clusters_vs[key_v][0]+1, clusters_vs[key_v][1], clusters_vs[key_v][2])

        if not key_h in clusters_hs:
            clusters_hs[key_h] = (1, ys, dist_h)
        elif dist_h < clusters_hs[key_h][2]:
            clusters_hs[key_h] = (clusters_hs[key_h][0]+1, ys, dist_h)
        else:
            clusters_hs[key_h] = (clusters_hs[key_h][0] + 1, clusters_hs[key_h][1], clusters_hs[key_h][2])


    clusters_v, n_max, xs, dist_v = [], 0, [], 0.0
    for cluster, (n, x, d) in clusters_vs.items():
        if n > n_max:
            n_max = n
            clusters_v = cluster
            xs = x
            dist_v = d

    clusters_h, n_max, ys, dist_h = [], 0, [], 0.0
    for cluster, (n, y, d) in clusters_hs.items():
        if n > n_max:
            n_max = n
            clusters_h = cluster
            ys = y
            dist_h = d

    return clusters_v, clusters_h, xs, ys, dist_v, dist_h


