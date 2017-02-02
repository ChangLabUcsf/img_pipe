import numpy as np
import scipy.io
def get_elecs_anat(subj_dir,subj,region):
    tdt = scipy.io.loadmat('%s/%s/elecs/TDT_elecs_all.mat'%(subj_dir,subj))
    return tdt['elecmatrix'][np.where(tdt['anatomy'][:,3]==region)[0],:]

def ctmr_plot(subj_dir,subj,hem, elecs, weights=None,interactive=False):
    #need ctmr_brain_plot, mayavi, vtk
    import ctmr_brain_plot
    from mayavi import mlab
    from matplotlib import cm
    import scipy.io
    import numpy as np
    import matplotlib.pyplot as plt
    
    a = scipy.io.loadmat('%s/%s/Meshes/%s_pial_trivert.mat'%(subj_dir, subj,hem));
    if weights==None:
        weights = np.ones((elecs.shape[0]))*-1.
    mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(a['tri'], a['vert'], elecs=elecs,weights=weights,color=(0.8, 0.8, 0.8),cmap='RdBu')

    mesh.actor.property.opacity = 1.0 # Make brain semi-transparent

    # View from the side
    if hem=='lh':
        azimuth=180
    elif hem=='rh':
        azimuth=0
    mlab.view(azimuth, elevation=90)
    arr = mlab.screenshot(antialiased=True)
    plt.figure(figsize=(20,10))
    plt.imshow(arr, aspect='equal')
    plt.axis('off')
    plt.show()
    if interactive:
        mlab.show()
    else:
        mlab.close()
    return mesh,mlab