import numpy as np

def sample_points(pcl, n_points):
    if(len(pcl) == n_points):
        return pcl
    if(len(pcl) > n_points):
        pcl_indx  = np.random.choice(len(pcl), n_points)
        return pcl[pcl_indx]
    if(len(pcl) < n_points):
        pcl_new = np.zeros((n_points,3))
        pcl_new[:len(pcl)] = pcl
        pcl_new[len(pcl):] = pcl[-1]
        return pcl_new
