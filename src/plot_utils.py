
import open3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def init_pc(v,f,colors, offset=[0,0,0]):
    pc = open3d.geometry.TriangleMesh()
    pc.vertices = open3d.utility.Vector3dVector(v + offset)
    pc.triangles = open3d.utility.Vector3iVector(f)
    pc.vertex_colors = open3d.utility.Vector3dVector(colors[:,0:3])
    # pc = open3d.geometry.PointCloud()
    # pc.points = open3d.utility.Vector3dVector(v + offset)
    # pc.colors = open3d.utility.Vector3dVector(colors[:,0:3])
    return pc


def plot_f(gt_v,v,f,func1, func2):
    print('plotting...')
    cmap = cm.get_cmap('Spectral')
    func1 = (func1 - np.min(func1)) / (np.max(func1) - np.min(func1))
    colors = cmap(func1)
    pc1 = init_pc(v,f,colors, offset = [1,0,0])

    func2 = (func2 - np.min(func2)) / (np.max(func2) - np.min(func2))
    colors = cmap(func2)
    pc2 = init_pc(gt_v, f, colors)
    open3d.visualization.draw_geometries([pc1, pc2])


def plot_f2(gt_v,v,f,func1, func2):
    print('plotting...')

    cmap = cm.get_cmap('Spectral')
    min_val = np.min([np.min(func1,0),np.min(func2,0)], 0)
    max_val = np.min([np.max(func1,0),np.max(func2,0)], 0)
    #func1 = (func1 - min_val)/np.tile((max_val-min_val),(np.size(func1,0),1))
    #func2 = (func2 - min_val)/np.tile((max_val-min_val),(np.size(func2,0),1))
    func1 = (func1 - min_val)/(np.max(func1) - np.min(func1))
    func2 = (func2 - min_val)/(np.max(func2) - np.min(func2))

    colors = np.transpose(np.stack([func1, np.ones(func1.shape[0])*0.5,np.ones(func1.shape[0])*0.5]))

    #colors = np.concatenate([func1, np.ones((func1.shape[0], 1))], 1)
    pc1 = init_pc(v,f,colors, offset = [1,0,0])
    colors = np.transpose(np.stack([func2, np.ones(func2.shape[0])*0.5,np.ones(func2.shape[0])*0.5]))
    #colors = np.concatenate([func2, np.ones((func1.shape[0], 1))], 1)
    pc2 = init_pc(gt_v, f, colors)
    open3d.visualization.draw_geometries([pc1, pc2])


def plot_corr(gt_v,v,f,corr):
    print('plotting...')
    cmap = cm.get_cmap('Spectral')
    min_val = np.min(gt_v,0)
    max_val = np.max(gt_v,0)
    func1 = (gt_v - min_val)/np.tile((max_val-min_val),(np.size(v,0),1))

    colors = np.concatenate([func1, np.ones((func1.shape[0], 1))], 1)
    pc1 = init_pc(v,f,colors, offset = [1,0,0])
    pc3 = init_pc(gt_v,f,colors, offset = [-1,0,0])

    colors = colors[corr,:]
    pc2 = init_pc(gt_v, f, colors)

    open3d.visualization.draw_geometries([pc1, pc2, pc3])
