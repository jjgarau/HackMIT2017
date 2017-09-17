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
#     best_kmeans
#     for i in range(3):
#         for j in range(3):



def Kmeans(vert_rects, horiz_rects, n_vert, n_horiz, max_coords, niter=10, ntimes=10):

    xs = np.random.uniform(low=0, high=max_coords[0], size=n_vert)
    ys = np.random.uniform(low=0, high=max_coords[1], size=n_horiz)

    clusters_hs = {}
    clusters_vs = {}

    for exec in ntimes:

        dist = 0.0
        clusters_h = np.zeros(shape=(len(horiz_rects), 2)) # asign each point to a cluster
        clusters_v = np.zeros(shape=(len(vert_rects), 2))
        for iter in niter:
            # maximization
            for i, x in enumerate(xs):
                d = vert_rects - x
                clusters_v[d < clusters_v[:,1], :] = [i, d]
            dist += np.sum(clusters_v[:,1] * clusters_v[:,1])

            for i, y in enumerate(ys):
                d = horiz_rects - y
                clusters_h[d < clusters_h[:,1], :] = [i, d]
            dist += np.sum(clusters_h[:,1] * clusters_h[:,1])

            # expectation
            xs = np.mean(vert_rects[clusters_v[:,0] == xs])
            ys = np.mean(horiz_rects[clusters_h[:,0] == ys])

        key_v = tuple(clusters_v[:,0])
        key_h = tuple(clusters_h[:,0])

        if not key_v in clusters_vs:
            clusters_v[key_v] = (0, 0.0)
        if not key_h in clusters_hs:
            clusters_h[key_h] = (0, 0.0)
        clusters_vs[key_v][0] += 1
        clusters_vs[key_v][1] += dist
        clusters_hs[key_h][0] += 1
        clusters_vs[key_v][1] += dist

    clusters_v, n_max, avg_dist_v = [], 0, 0.0
    for cluster, (n, d) in clusters_vs.items():
        if n > n_max:
            n_max = n
            clusters_v = cluster
            avg_dist_v = d *1. / n

    clusters_h, n_max, avg_dist_h = [], 0, 0.0
    for cluster, (n, d) in clusters_hs.items():
        if n > n_max:
            n_max = n
            clusters_h = cluster
            avg_dist_h = d * 1. / n

    return clusters_v, clusters_h, avg_dist_v, avg_dist_h


