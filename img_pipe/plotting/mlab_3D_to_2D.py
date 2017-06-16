# mlab_3D_to_2D tools
#
# from example_mlab_3D_to_2D by S. Chris Colbert <sccolbert@gmail.com>, 
# see: http://docs.enthought.com/mayavi/mayavi/auto/example_mlab_3D_to_2D.html
#

import numpy as np

def get_world_to_view_matrix(mlab_scene):
    """returns the 4x4 matrix that is a concatenation of the modelview transform and
    perspective transform. Takes as input an mlab scene object.
    
    Parameters
    ----------
    mlab_scene : MayaviScene instance

    Returns
    -------
    np_comb_trans_mat : array-like
        Combined 4x4 transformation matrix for modelview and perspective transforms
    """
    from mayavi.core.ui.mayavi_scene import MayaviScene

    if not isinstance(mlab_scene, MayaviScene):
        raise TypeError('argument must be an instance of MayaviScene')


    # The VTK method needs the aspect ratio and near and far clipping planes
    # in order to return the proper transform. So we query the current scene
    # object to get the parameters we need.
    scene_size = tuple(mlab_scene.get_size())
    clip_range = mlab_scene.camera.clipping_range
    aspect_ratio = float(scene_size[0])/float(scene_size[1])

    # this actually just gets a vtk matrix object, we can't really do anything with it yet
    vtk_comb_trans_mat = mlab_scene.camera.get_composite_projection_transform_matrix(
                                aspect_ratio, clip_range[0], clip_range[1])

     # get the vtk mat as a numpy array
    np_comb_trans_mat = vtk_comb_trans_mat.to_array()

    return np_comb_trans_mat

def get_view_to_display_matrix(mlab_scene):
    """ this function returns a 4x4 matrix that will convert normalized
        view coordinates to display coordinates. It's assumed that the view should
        take up the entire window and that the origin of the window is in the
        upper left corner
    
    Parameters
    ----------
    mlab_scene : MayaviScene instance

    Returns
    -------
    view_to_disp_mat : array-like
        4 x 4 matrix that will convert normalized view coordinates to display coordinates.
    
    """
    from mayavi.core.ui.mayavi_scene import MayaviScene

    if not (isinstance(mlab_scene, MayaviScene)):
        raise TypeError('argument must be an instance of MayaviScene')

    # this gets the client size of the window
    x, y = tuple(mlab_scene.get_size())
    print(x,y)

    # normalized view coordinates have the origin in the middle of the space
    # so we need to scale by width and height of the display window and shift
    # by half width and half height. The matrix accomplishes that.
    view_to_disp_mat = np.array([[x/2.0,      0.,   0.,   x/2.0],
                                 [   0.,  -y/2.0,   0.,   y/2.0],
                                 [   0.,      0.,   1.,      0.],
                                 [   0.,      0.,   0.,      1.]])

    return view_to_disp_mat

def apply_transform_to_points(points, trans_mat):
    """a function that applies a 4x4 transformation matrix to an of
        homogeneous points. The array of points should have shape Nx4
    
    Parameters
    ----------
    points : array-like
        Nx4 matrix of points in homogeneous coordinates
    trans_mat : array-like
        4x4 transformation matrix

    Returns
    -------
    tpoints : array-like
        Transformed point matrix 

    """

    if not trans_mat.shape == (4, 4):
        raise ValueError('transform matrix must be 4x4')

    if not points.shape[1] == 4:
        raise ValueError('point array must have shape Nx4')

    tpoints = np.dot(trans_mat, points.T).T
    return tpoints