# plot_from_nwb.py
# author: Luiz Tauffer
# date: 20.04.2019

import matplotlib.pyplot as plt
import mayavi
import scipy.io
from .ctmr_brain_plot import ctmr_gauss_plot, el_add
from img_pipe.SupplementalFiles import FS_colorLUT as FS_colorLUT
import numpy as np
import os

import pynwb
import nwbext_ecog

fs_dir = os.environ['SUBJECTS_DIR']

def plot_from_nwb(subj_file,  
                  fs_dir=fs_dir, 
                  hem='stereo',
                  opacity=1.0,
                  roi = [],
                  electrodes=False,
                  elec_roi=[],
                  elec_space='original',
		  interactive=False):
    '''
    This function will plot the mesh structures and electrode positions 
    contained in a NWB file.

    Parameters
    ----------
    subj_file : str
        subject file's name
    fs_dir : str, optional
        subject file's directory. Default is Freesurfer's $SUBJECTS_DIR
        environment variable.
    hem : str, optional
        hemisphere: 'lh', 'rh' or 'stereo'. Default='stereo'
    opacity : float, optional
        opacity of the mesh, between 0.0 and 1.0. Default=1.0
    roi : list, optional
        list of ROI (dictionaries) to be highlighted. Default=[]
    electrodes : boolean, optional
        if True plots electrodes. Default=False
    elec_roi : list, optional
        list of ROI of electrodes to plot. Default=[]
    elec_space : str, optional
        'original' or 'warped'. Default='original'
    

    Returns
    -------
    mesh : mayavi brain mesh
    mlab : mayavi mlab scene
    None : if an error occurred
    '''
    
    fpath = os.path.join(fs_dir, subj_file)
    
    # Check for file path
    exists = os.path.isfile(fpath)
    if exists:
        io = pynwb.NWBHDF5IO(fpath,'r')
        nwb = io.read()
    else:
        print('ERROR: File or path does not exist!')
        print('Given path to file: ', fpath)
        print('The standard path is given by your local Freesurfer installation',
              '$SUBJECTS_DIR variable. If this is a wrong path, put the correct',
              'path as an argument, e.g.:')
        print(' plot_recon_anatomy_nwb(subj_file, hem, fs_dir=path_to_file) ')
        return None
    
    # Check for Brain Atlas
    if elec_space!='original':
        # Check for file path
        fpath_atlas = os.path.join(fs_dir, 'cvs_avg35_inMNI152.nwb')
        exists = os.path.isfile(fpath_atlas)
        if exists:
            io_atlas = pynwb.NWBHDF5IO(fpath_atlas,'r')
            nwb_atlas = io_atlas.read()
        else:
            print('ERROR: Could not find Brain Atlas!')
            print('Searched path: ', fpath_atlas)
            print('The standard path is given by your local Freesurfer installation',
                  '$SUBJECTS_DIR variable. Please save the Atlas file in this folder')
            return None
    
    # Subject ID
    subject_id = nwb.subject.subject_id
    
    # Get mesh and plot the pial surfaces
    if (hem=='lh') or (hem=='rh'):
        if elec_space=='original':
            #pial_mesh = nwb.subject.cortical_surfaces.surfaces[subject_id+'_'+hem+'_pial']
            pial_mesh = nwb.subject.cortical_surfaces.surfaces[hem]
        elif elec_space=='warped':
            pial_mesh = nwb_atlas.subject.cortical_surfaces.surfaces[hem]
        else:
            print('ERROR: Invalid elec_space name.')
            return None
        mesh, mlab = ctmr_gauss_plot(pial_mesh.faces[()], 
                                     pial_mesh.vertices[()],
                                     opacity=opacity,
                                     color=(0.8, 0.8, 0.8))
        
    elif hem=='stereo':
        if elec_space=='original':
            #pial_mesh_l = nwb.subject.cortical_surfaces[subject_id+'_lh_pial']
            pial_mesh_l = nwb.subject.cortical_surfaces['lh']
            #pial_mesh_r = nwb.subject.cortical_surfaces[subject_id+'_rh_pial']
            pial_mesh_r = nwb.subject.cortical_surfaces['rh']
        elif elec_space=='warped':
            pial_mesh_l = nwb_atlas.subject.cortical_surfaces.surfaces['lh']
            pial_mesh_r = nwb_atlas.subject.cortical_surfaces.surfaces['rh']
        else:
            print('ERROR: Invalid elec_space name.')
            return None
        
        mesh, mlab = ctmr_gauss_plot(pial_mesh_l.faces[()], 
                                     pial_mesh_l.vertices[()],  
                                     opacity=opacity,
                                     color=(0.8, 0.8, 0.8))
        
        mesh, mlab = ctmr_gauss_plot(pial_mesh_r.faces[()], 
                                     pial_mesh_r.vertices[()],  
                                     opacity=opacity,
                                     color=(0.8, 0.8, 0.8),
                                     new_fig=False)
    
    # all anatomic regions
    anatomic_regions = list(nwb.subject.cortical_surfaces.surfaces.keys())
    
    # Paint chosen ROI in distinguished colors
    for ind, rg in enumerate(roi):
        #full_name = subject_id+'_'+rg['name']+'_pial'
        full_name = rg['name']
        if full_name in anatomic_regions:
            rg_color = rg['color']
            rg_mesh = nwb.subject.cortical_surfaces.surfaces[full_name]
            
            mesh, mlab = ctmr_gauss_plot(rg_mesh.faces[()], 
                                         rg_mesh.vertices[()],  
                                         new_fig=False,
                                         color=rg_color)
        else:
            print(full_name,' not present in the provided data file!')
            print('Check list of available parcellated areas with:')
            print('  list(nwb.subject.cortical_surfaces.surfaces.keys())')
        

    
    # Plot electrodes
    if electrodes:
        # Electrodes positions
        if elec_space=='original':
            x_pos = nwb.electrodes['x'][:]
            y_pos = nwb.electrodes['y'][:]
            z_pos = nwb.electrodes['z'][:]
            elec_pos = np.zeros((x_pos.shape[0],3)) 
            elec_pos[:,0] = x_pos
            elec_pos[:,1] = y_pos
            elec_pos[:,2] = z_pos
        elif elec_space=='warped':
            x_pos = nwb.electrodes['x_warped'][:]
            y_pos = nwb.electrodes['y_warped'][:]
            z_pos = nwb.electrodes['z_warped'][:]
            elec_pos = np.zeros((x_pos.shape[0],3)) 
            elec_pos[:,0] = x_pos
            elec_pos[:,1] = y_pos
            elec_pos[:,2] = z_pos
        
        region = nwb.electrodes.columns[4][:]
    
        # Import freesurfer color lookup table as a dictionary
        cmap = FS_colorLUT.get_lut()
        
        # Make a list of electrode numbers
        elec_numbers = np.arange(x_pos.shape[0])+1
    
        # Find all the unique brain areas in this subject
        region = nwb.electrodes.columns[4][:]
        all_regions = list(set(region))
        
        # If no ROIs were chosen for electrodes
        if len(elec_roi)==0:
            elec_roi = all_regions
        
        # Loop through electrodes and plot them in each brain area
        for ind, pos in enumerate(elec_pos):
            b = region[ind]
            if b[0:3]!='ctx' and b[0:4] != 'Left' and b[0:5] != 'Right' and b[0:5] != 'Brain' and b != 'Unknown':
                if x_pos[ind]<0: #left side
                    this_label = 'ctx-lh-%s'%(b)
                else:           #right side
                    this_label = 'ctx-rh-%s'%(b)
                    
            if (this_label != '') and (b in elec_roi):
                try:    #check if 'this_label' exists within cmap
                    el_color = np.array(cmap[this_label])/255.
                    el_add(np.atleast_2d(pos), color=tuple(el_color))
                except:
                    el_add(np.atleast_2d(pos))
       
    if interactive:
        mlab.show()
    #else:
    #    mlab.close()
    return mesh, mlab
