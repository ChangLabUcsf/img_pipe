# ctmr_brain_plot.py                                                                                                 
''' This module contains a function (ctmr_brain_plot) that takes as 
 input a 3d coordinate set of triangular mesh vertices (vert) and 
 an ordered list of their indices (tri), to produce a 3d surface 
 model of an individual brain. Assigning the result of the plot 
 to a variable enables interactive changes to be made to the OpenGl 
 mesh object. Default shading is phong point shader (shiny surface).

 usage: from ctmr_brain_plot import *
        dat = scipy.io.loadmat('/path/to/lh_pial_trivert.mat');
        mesh = ctmr_brain_plot(dat['tri'], dat['vert']);
        mlab.show()

 A second function contained in this module can be used to plot electrodes
 as glyphs (spehres) or 2d circles. The function (el_add) takes as input
 a list of 3d coordinates in freesurfer surface RAS space and plots them
 according to the color and size parameters that you provide.

 usage: elecs = scipy.io.loadmat('/path/to/hd_grid.mat')['elecmatrix'];
        points = el_add(elecs, color = (1, 0, 0), msize = 2.5);
        mlab.show()                                                     
'''

import scipy.io
import mayavi
from mayavi import mlab
import numpy as np

def ctmr_gauss_plot(tri, vert, color = (0.6, 0.6, 0.6), elecs = [], weights = [], 
                    opacity = 1.0, representation = 'surface', line_width=1.0, gsp = 10,
                    cmap = 'RdBu'):
    '''
    ctmr_gauss_plot(tri, vert)
    This function plots the 3D brain surface mesh
    Inputs:
        color: (n,n,n) tuple of floats between 0.0 and 1.0, background color of brain
        elecs: [nchans x 3] matrix of electrode coordinate values in 3D
        weights: [nchans x 1] - if [elecs] is also given, this will color the brain vertices according to these weights
        msize: size of the electrode.  default = 2
        opacity: opacity of the brain surface (value from 0.0 - 1.0)
        cmap: [str], colormap to use when plotting gaussian weights with [elecs] and [weights]
        representation: 'surface' (default), or 'wireframe'
        line_width: [float]
        gsp: [int], gaussian smoothing parameter
    '''

    brain_color = []
    #c = np.zeros(vert.shape[0],)

    if elecs!=[]:
        brain_color = np.zeros(vert.shape[0],)
        for i in np.arange(elecs.shape[0]):
            b_z = np.abs(vert[:,2] - elecs[i,2])
            b_y = np.abs(vert[:,1] - elecs[i,1])
            b_x = np.abs(vert[:,0] - elecs[i,0])
            gauss_wt = np.nan_to_num(weights[i] * np.exp((-(b_x**2+b_z**2+b_y**2))/gsp)) #gaussian
            brain_color = brain_color + gauss_wt
    print brain_color

    # plot cortex and begin display
    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200,900))

    if elecs!=[]:
        mesh = mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2], tri, 
                                representation = representation, 
                                opacity = opacity, line_width = line_width, scalars=brain_color,
                                colormap = cmap, vmin=-np.abs(brain_color).max(), vmax=np.abs(brain_color).max())
    else:
        mesh = mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2], tri, 
                                color=color, 
                                representation = representation, 
                                opacity = opacity, line_width = line_width)

    # cell_data = mesh.mlab_source.dataset.cell_data
    # cell_data.scalars = brain_color
    # cell_data.scalars.name = 'Cell data'
    # cell_data.update()

    #mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars = 'Cell data')
    #mlab.pipeline.surface(mesh)

    # change OpenGL mesh properties for phong point light shading
    mesh.actor.property.ambient = 0.4225
    mesh.actor.property.specular = 0.333
    mesh.actor.property.specular_power = 66
    mesh.actor.property.diffuse = 0.6995
    mesh.actor.property.interpolation = 'phong'
    mesh.scene.light_manager.light_mode = 'vtk'
    return mesh, mlab


def el_add(elecs, color = (1., 0., 0.), msize = 2, numbers = None):
    '''
    el_add(elecs, color = (1., 0., 0.), msize = 2)
    This function adds the electrode matrix [elecs] (nchans x 3) to 
    the scene.  
    Inputs:
        elecs: [nchans x 3] matrix of electrode coordinate values in 3D
        color: Electrode color is either a triplet (r, g, b),
               or a numpy array with the same shape as [elecs] to plot one color per electrode
        msize: size of the electrode.  default = 2
    '''

    # plot the electrodes as spheres
    # If we have one color for each electrode, color them separately
    if type(color) is np.ndarray:
        if color.shape[0] == elecs.shape[0]: 
            for e in np.arange(elecs.shape[0]):
                points = mlab.points3d(elecs[e,0], elecs[e,1], elecs[e,2], scale_factor = msize, 
                                   color = tuple( color[e,:] ) , resolution=25)
        else:
            print('Warning: color array does not match size of electrode matrix')

    # Otherwise, use the same color for all electrodes
    else:
        points = mlab.points3d(elecs[:,0],elecs[:,1], elecs[:,2], scale_factor = msize, color = color, resolution=25)
    
    # Set display properties
    points.actor.property.ambient = 0.3261
    points.actor.property.specular = 1
    points.actor.property.specular_power = 16
    points.actor.property.diffuse = 0.6995
    points.actor.property.interpolation = 'phong'
    points.scene.light_manager.light_mode = 'vtk'

    if numbers is not None:
        for ni, n in enumerate(numbers):
            mayavi.mlab.text3d(elecs[ni,0], elecs[ni,1], elecs[ni,2], str(n))

    return points, mlab