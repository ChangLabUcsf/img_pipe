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

Modified for use in python from MATLAB code originally written by 
Kai Miller and Dora Hermes (ctmr_gui, see https://github.com/dorahermes/Paper_Hermes_2010_JNeuroMeth)

'''

import scipy.io
import mayavi
from mayavi import mlab
import numpy as np
import matplotlib as mpl

def ctmr_gauss_plot(tri, vert, color=(0.8, 0.8, 0.8), elecs=None, weights=None,
                    opacity = 1.0, representation='surface', line_width=1.0, gsp = 10,
                    cmap=mpl.cm.get_cmap('RdBu_r'), show_colorbar=True, new_fig=True, vmin=None, vmax=None,
                    ambient=0.4225, specular = 0.333, specular_power = 66, diffuse = 0.6995, interpolation='phong'):
    ''' This function plots the 3D brain surface mesh
    
    Parameters
    ----------
        color : tuple
            (n,n,n) tuple of floats between 0.0 and 1.0, background color of brain
        elecs : array-like
            [nchans x 3] matrix of electrode coordinate values in 3D
        weights : array-like 
            [nchans x 1] - if [elecs] is also given, this will color the brain vertices 
            according to these weights
        msize : float
            size of the electrode.  default = 2
        opacity : float (0.0 - 1.0)
            opacity of the brain surface (value from 0.0 - 1.0)
        cmap : str or mpl.colors.LinearSegmentedColormap
            colormap to use when plotting gaussian weights with [elecs]
            and [weights]
        representation : {'surface', 'wireframe'}
            surface representation
        line_width : float
            width of lines for triangular mesh
        gsp : float
            gaussian smoothing parameter, larger makes electrode activity
            more spread out across the surface if specified

    '''
    # if color is another iterable, make it a tuple.
    color = tuple(color)

    brain_color = []
    #c = np.zeros(vert.shape[0],)

    if elecs is not None:
        brain_color = np.zeros(vert.shape[0],)
        for i in np.arange(elecs.shape[0]):
            b_z = np.abs(vert[:, 2] - elecs[i, 2])
            b_y = np.abs(vert[:, 1] - elecs[i, 1])
            b_x = np.abs(vert[:, 0] - elecs[i, 0])
            gauss_wt = np.nan_to_num(weights[i] * np.exp((-(b_x**2+b_z**2+b_y**2))/gsp)) #gaussian
            brain_color = brain_color + gauss_wt

        #scale the colors so that it matches the weights that were passed in
        brain_color = brain_color * (np.abs(weights).max()/np.abs(brain_color).max())
        if vmin==None and vmax==None:
            vmin, vmax = -np.abs(brain_color).max(), np.abs(brain_color).max()

    # plot cortex and begin display
    if new_fig:
        mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200,900))

    if elecs is not None:
        kwargs = {}
        if type(cmap) == str:
            kwargs.update(colormap=cmap)

        mesh = mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], tri,
                                    representation=representation, opacity=opacity,
                                    line_width=line_width, scalars=brain_color,
                                    vmin=vmin, vmax=vmax, **kwargs)

        if type(cmap) == mpl.colors.LinearSegmentedColormap:
            mesh.module_manager.scalar_lut_manager.lut.table = (cmap(np.linspace(0, 1, 255)) * 255).astype('int')
    else:
        mesh = mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], tri,
                                color=color, representation=representation,
                                opacity=opacity, line_width=line_width)

    # cell_data = mesh.mlab_source.dataset.cell_data
    # cell_data.scalars = brain_color
    # cell_data.scalars.name = 'Cell data'
    # cell_data.update()

    #mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars = 'Cell data')
    #mlab.pipeline.surface(mesh)
    if weights is not None and show_colorbar:
        mlab.colorbar()

    # change OpenGL mesh properties for phong point light shading
    mesh.actor.property.ambient = ambient
    mesh.actor.property.specular = specular
    mesh.actor.property.specular_power = specular_power
    mesh.actor.property.diffuse = diffuse
    mesh.actor.property.interpolation = interpolation
    mesh.scene.light_manager.light_mode = 'vtk'
    if opacity < 1.0:
        mesh.scene.renderer.set(use_depth_peeling=True) #, maximum_number_of_peels=100, occlusion_ratio=0.0

    return mesh, mlab


def el_add(elecs, color = (1., 0., 0.), msize = 2, numbers = None, label_offset=-1.0, ambient = 0.3261, specular = 1, specular_power = 16, diffuse = 0.6995, interpolation = 'phong'):
    '''This function adds the electrode matrix [elecs] (nchans x 3) to
    the scene.
    
    Parameters
    ----------
        elecs : array-like
            [nchans x 3] matrix of electrode coordinate values in 3D
        color : tuple (triplet) or numpy array
            Electrode color is either a triplet (r, g, b),
            or a numpy array with the same shape as [elecs] to plot one color per electrode
        msize : float
            size of the electrode.  default = 2
        label_offset : float
            how much to move the number labels out by (so not blocked by electrodes)
    '''

    # plot the electrodes as spheres
    # If we have one color for each electrode, color them separately
    if type(color) is np.ndarray:
        if color.shape[0] == elecs.shape[0]:
            # for e in np.arange(elecs.shape[0]):
            #     points = mlab.points3d(elecs[e,0], elecs[e,1], elecs[e,2], scale_factor = msize,
            #                        color = tuple( color[e,:] ) , resolution=25)
            unique_colors = np.array(list(set([tuple(row) for row in color])))
            for individual_color in unique_colors:
                indices = np.where((color==individual_color).all(axis=1))[0]
                points = mlab.points3d(elecs[indices,0],elecs[indices,1],elecs[indices,2],scale_factor=msize,color=tuple(individual_color),resolution=25)
        else:
            print('Warning: color array does not match size of electrode matrix')

    # Otherwise, use the same color for all electrodes
    else:
        points = mlab.points3d(elecs[:,0],elecs[:,1], elecs[:,2], scale_factor = msize, color = color, resolution=25)

    # Set display properties
    points.actor.property.ambient = ambient
    points.actor.property.specular = specular
    points.actor.property.specular_power = specular_power
    points.actor.property.diffuse = diffuse
    points.actor.property.interpolation = interpolation
    points.scene.light_manager.light_mode = 'vtk'

    if numbers is not None:
        for ni, n in enumerate(numbers):
            mayavi.mlab.text3d(elecs[ni,0]+label_offset, elecs[ni,1], elecs[ni,2], str(n), orient_to_camera=True) #line_width=5.0, scale=1.5)

    return points, mlab
