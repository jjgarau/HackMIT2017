import numpy as np


def orient_vert(theta):
    if -5*np.pi/180 < theta < 5*np.pi/180:
        return False
    else:
        return True


def get_orient_xy(thetas, dists):
    o_v = map(orient_vert, thetas)
    verts = dists*np.cos(thetas)[o_v]
    horizs = dists*np.sin(thetas)[np.logical_not(o_v)]
    return verts, horizs


# rects_td is tuple of thetas, dist
# def find_best_Kmeans(rects_td, max_coords):
#     verts, horizs = get_orient_xy(rects_td)
#     best_kmeans = [], [], [], [], np.inf, np.inf
#     for i in range(3):
#         for j in range(3):
#             clusters_v, clusters_h, xs, ys, dist_v, dist_h


def Kmeans(vert_rects, horiz_rects, n_vert, n_horiz, max_coords, niter=10, ntimes=10):

    clusters_hs = {}
    clusters_vs = {}

    for exec in range(ntimes):

        xs = np.random.uniform(low=0, high=max_coords[0], size=n_vert)
        ys = np.random.uniform(low=0, high=max_coords[1], size=n_horiz)

        print(xs.shape)
        print(ys.shape)

        dist_v = 0.0
        dist_h = 0.0
        clusters_h = [0] * horiz_rects # asign each point to a cluster
        dist_clust_h = np.array([np.inf] * len(horiz_rects))
        clusters_v = [0] * vert_rects
        dist_clust_v = np.array([np.inf] * len(vert_rects))

        for k in range(niter):
            # maximization
            for i, x in enumerate(xs):
                d = np.abs(vert_rects - x)
                print(d)
                for j in range(len(d)):
                    if d[j] < dist_clust_v[j]:
                        print(d[j], dist_clust_v[j])
                        clusters_v[j] = i
                        dist_clust_v[j] = d[j]
            dist_v += np.sum(dist_clust_v * dist_clust_v)

            for i, y in enumerate(ys):
                d = horiz_rects - y
                for j in range(len(d)):
                    if d[j] < dist_clust_h[j]:
                        clusters_h[j] = i
                        dist_clust_h[j] = d[j]
            dist_h += np.sum(dist_clust_h * dist_clust_h)

            print(clusters_v)
            print(clusters_h)
            # expectation
            for clust in clusters_v:
                xs[clust] = np.mean(vert_rects[clusters_v == clust])
            for clust in clusters_h:
                ys[clust] = np.mean(horiz_rects[clusters_h == clust])

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
            clusters_vs[key_h] = (clusters_hs[key_h][0] + 1, clusters_hs[key_h][1], clusters_hs[key_h][2])

    print(clusters_vs)
    print(clusters_hs)

    clusters_v, n_max, xs, dist_v = [], 0, [], 0.0
    for cluster, (n, x, d) in clusters_vs.items():
        if n > n_max:
            n_max = n
            clusters_v = cluster
            xs = x
            dist_v = d

    clusters_h, n_max, ys, avg_dist_h = [], 0, [], 0.0
    for cluster, (n, y, d) in clusters_hs.items():
        if n > n_max:
            n_max = n
            clusters_h = cluster
            ys = y
            dist_h = d

    return clusters_v, clusters_h, xs, ys, dist_v, dist_h


