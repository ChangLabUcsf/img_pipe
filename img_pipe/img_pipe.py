# FreeCoG Imaging Pipeline 
# Developed by:
# Liberty Hamilton, Morgan Lee, David Chang, Anthony Fong, Zachary Greenberg, Ben Speidel
# 
# Laboratory of Edward Chang
# Department of Neurological Surgery
# University of California, San Francisco
# Date Last Edited: March 15, 2019
#
# This file contains the Chang Lab imaging pipeline (freeCoG)
# as one importable python class for running a patients
# brain surface reconstruction and electrode localization/labeling

import os
import glob
import pickle
import shutil
import argparse
import inspect

import nibabel as nib
from tqdm import tqdm

import numpy as np
import scipy.io
import scipy.spatial
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

# For CT to MRI registration
from nipy.core.api import AffineTransform
import nipy.algorithms
import nipy.algorithms.resample
import nipy.algorithms.registration.histogram_registration

# For reading and unpacking binary files
import struct

from .plotting.mlab_3D_to_2D import get_world_to_view_matrix, get_view_to_display_matrix, apply_transform_to_points

# For animations, from pycortex
linear = lambda x, y, m: (1.-m)*x + m*y
mixes = dict(
    linear=linear,
    smoothstep=(lambda x, y, m: linear(x,y,3*m**2 - 2*m**3)),
    smootherstep=(lambda x, y, m: linear(x, y, 6*m**5 - 15*m**4 + 10*m**3))
)

class freeCoG:
    ''' This defines the class freeCoG, which creates a patient object      
    for use in creating brain surface reconstructions, electrode placement,     
    and warping.        
    
    To initialize a patient, you must provide the subject ID, hemisphere,       
    freesurfer subjects directory, and (optionally) the freesurfer      
    executable directory. If these aren't specified they will default to the environment
    variables $SUBJECTS_DIR and $FREESURFER_HOME.
    
    For example:        
    
    >>> subj = 'EC1'     
    >>> subj_dir = '/usr/local/freesurfer/subjects'      
    >>> hem = 'rh'       
    >>> fs_dir = '/usr/local/freesurfer'     
    >>> patient = img_pipe.freeCoG(subj = subj, subj_dir = subj_dir, hem = hem, fs_dir = fs_dir)
    
    Parameters
    ----------       
    subj : str 
           The subject ID      
    hem : {'lh', 'rh', 'stereo'}
          The hemisphere of implanation. Can be 'lh', 'rh', or 'stereo'.        
    zero_indexed_electrodes: bool, optional
                             Whether or not to use zero-indexing for the electrode numbers (default: True)
    fs_dir : str, optional
             Path to the freesurfer install (default is $FREESURFER_HOME environmental variable) 
    subj_dir : str, optional
               the freesurfer subjects directory (e.g. /usr/local/freesurfer/subjects). Default
               is the $SUBJECTS_DIR environmental variable set by Freesurfer.
    
    Attributes
    ----------
    subj : str
        The subject ID.
    subj_dir : str
        The subjects directory where the data live.
    patient_dir : str
        The directory containing data for this particular patient.
        Usually [subj_dir]/[subj]
    hem : str
        The hemisphere of implantation
    img_pipe_dir : str
        The path to img_pipe code.
    zero_indexed_electrodes : bool
        Whether zero-indexed electrode numbers are used (True)
        or not (False)
    fs_dir : str
        Path to the freesurfer installation
    CT_dir : str
        Path to the CT scan
    elecs_dir : str
        Path to the marked electrode file locations
    mesh_dir : str
        Path to the generated triangle-mesh surfaces (pial, subcortical, etc.)
    pial_surf_file : dict
        Dictionary containing full file with path for the left and right pial surfaces.
        Left pial surface is pial_surf_file['lh'] and right pial surface is
        pial_surf_file['rh']
    surf_dir : str
        The freesurfer surf directory for this patient.
    mri_dir : str
        The freesurfer MRI directory for this patient.

    Returns
    ----------
    patient : <img_pipe.freeCoG instance>
        patient object, including information about subject ID, hemisphere, where data live, 
        and related functions for creating surface reconstructions and/or plotting.    

    '''

    def __init__(self, subj, hem, zero_indexed_electrodes=True, fs_dir=os.environ['FREESURFER_HOME'], subj_dir=os.environ['SUBJECTS_DIR']):
        '''
        Initializes the patient object.

        Parameters
        ----------       
        subj : str 
            The subject ID (e.g. 'SUBJ_25')     
        hem : {'lh', 'rh', 'stereo'}
            The hemisphere of implanation. Can be 'lh', 'rh', or 'stereo'.        
        zero_indexed_electrodes: bool, optional
            Whether or not to use zero-indexing for the electrode numbers (default: True)
        fs_dir : str, optional
            Path to the freesurfer install (default is $FREESURFER_HOME environmental variable) 
        subj_dir : str, optional
            Path to the freesurfer subjects (default is $SUBJECTS_DIR environmental variable)
    
        Returns
        ----------   
        patient : <img_pipe.freeCoG instance>
            patient object, including information about subject ID, hemisphere, where data live, 
            and related functions for creating surface reconstructions and/or plotting.    
        '''
        # Check if hem is valid
        if not hem in ['rh', 'lh', 'stereo']:
            raise NameError('Invalid hem for freeCoG')
        
        self.subj = subj
        self.subj_dir = subj_dir
        self.patient_dir = os.path.join(self.subj_dir, self.subj)
        self.hem = hem
        self.img_pipe_dir = os.path.dirname(os.path.realpath(__file__))
        self.zero_indexed_electrodes = zero_indexed_electrodes

        # Freesurfer home directory
        self.fs_dir = fs_dir

        # acpc_dir: dir for acpc MRIs
        self.acpc_dir = os.path.join(self.subj_dir, self.subj, 'acpc')

        # CT_dir: dir for CT img data
        self.CT_dir = os.path.join(self.subj_dir, self.subj, 'CT')

        # dicom_dir: dir for storing dicoms
        self.dicom_dir = os.path.join(self.subj_dir, self.subj, 'dicom')

        # elecs_dir: dir for elecs coordinates
        self.elecs_dir = os.path.join(self.subj_dir, self.subj, 'elecs')

        # Meshes directory for matlab/python meshes
        self.mesh_dir = os.path.join(self.subj_dir, self.subj, 'Meshes')
        if self.hem == 'stereo':
            surf_file = os.path.join(self.subj_dir, self.subj, 'Meshes', 'lh_pial_trivert.mat')
        else:
            surf_file = os.path.join(self.subj_dir, self.subj, 'Meshes', self.hem + '_pial_trivert.mat')
        if os.path.isfile(surf_file):
            self.pial_surf_file = dict()
            self.pial_surf_file['lh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'lh_pial_trivert.mat')
            self.pial_surf_file['rh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'rh_pial_trivert.mat')

        # surf directory
        self.surf_dir = os.path.join(self.subj_dir, self.subj, 'surf')

        # mri directory
        self.mri_dir = os.path.join(self.subj_dir, self.subj, 'mri')

        # Check if subj is valid
        if not os.path.isdir(os.path.join(self.subj_dir, self.subj)):
            print('No such subject directory as %s' % (os.path.join(self.subj_dir, self.subj)))
            ans = raw_input('Would you like to create a new subject directory? (y/N): ')
            if ans.upper() == 'Y' or ans.upper() == 'YES':
                os.chdir(self.subj_dir)
                print("Making subject directory")
                os.mkdir(self.subj)
                os.mkdir(self.acpc_dir)
                os.mkdir(self.CT_dir)
                os.mkdir(self.dicom_dir)
                os.mkdir(self.elecs_dir)
                os.mkdir(self.mesh_dir)
                os.mkdir(self.mri_dir)
            else:
                raise NameError('No such subject directory as %s' % (os.path.join(subj_dir, subj)))

        #if paths are not the default paths in the shell environment:
        os.environ['FREESURFER_HOME'] = fs_dir
        os.environ['SUBJECTS_DIR'] = subj_dir

    def prep_recon(self):
        '''Prepares file directory structure of subj_dir, copies acpc-aligned               
        T1.nii to the 'orig' directory and converts to mgz format.

        '''       

        # navigate to directory with subject freesurfer folders           
        # and make a new folder for the patient

        # create elecs folders
        if not os.path.isdir(self.elecs_dir):
            print("Making electrodes directory")
            os.mkdir(self.elecs_dir)

        # create mri and orig folders
        if not os.path.isdir(self.mri_dir):
            os.mkdir(self.mri_dir)

        orig_dir = os.path.join(self.subj_dir, self.subj, 'mri', 'orig')
        if not os.path.isdir(orig_dir):
            os.mkdir(orig_dir)

        T1_file = os.path.join(self.subj_dir, self.subj, 'acpc', 'T1.nii')
        os.system('cp %s %s' %(T1_file, orig_dir))

        # convert T1 to freesurfer 001.mgz format
        T1_file2 = os.path.join(orig_dir, 'T1.nii')
        T1_mgz = os.path.join(orig_dir, '001.mgz')
        os.system('mri_convert %s %s'%(T1_file2, T1_mgz))

    def get_recon(self, flag_T3 = '-3T', openmp_flag='-openmp 12', gpu_flag=''):        
        '''Runs freesurfer recon-all for surface reconstruction.                
        
        Parameters
        ----------
        flag_T3 : {'-3T', ''}
            Choose '-3T' if using scans from a 3T scanner, otherwise set flag_T3=''             
        openmp_flag : {'-openmp 12', ''}
            First option uses 12 cores and multithreading to make recon-all faster. 
            Otherwise use openmp_flag = ''        
        gpu_flag : {'-use-gpu', ''}
            Use flag '-use-gpu' if you want to run code on the GPU (some steps run faster)        
            otherwise use gpu_flag='' 

        '''       
        os.system('recon-all -subjid %s -sd %s -all %s %s %s' % (self.subj, self.subj_dir, flag_T3, openmp_flag, gpu_flag))

        self.pial_surf_file = dict()
        self.pial_surf_file['lh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'lh_pial_trivert.mat')
        self.pial_surf_file['rh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'rh_pial_trivert.mat')

        #create gyri labels directory
        gyri_labels_dir = os.path.join(self.subj_dir, self.subj, 'label', 'gyri')
        if not os.path.isdir(gyri_labels_dir):
            os.mkdir(gyri_labels_dir)
            # This version of mri_annotation2label uses the coarse labels from the Desikan-Killiany Atlas
            os.system('mri_annotation2label --subject %s --hemi %s --surface pial --outdir %s'\
                %(self.subj, 'lh', gyri_labels_dir))
            os.system('mri_annotation2label --subject %s --hemi %s --surface pial --outdir %s'\
                %(self.subj, 'rh', gyri_labels_dir))

    def check_pial(self):
        '''Opens Freeview with the orig.mgz MRI loaded along with the pial surface. 
        User should scroll through to check that the pial surface corresponds correctly
        to the MRI.'''
        brain_mri = os.path.join(self.subj_dir, self.subj, 'mri', 'brain.mgz')
        lh_pial = os.path.join(self.subj_dir, self.subj, 'surf', 'lh.pial')
        rh_pial = os.path.join(self.subj_dir, self.subj, 'surf', 'rh.pial')
        os.system("freeview --volume %s --surface %s --surface %s --viewport 'coronal'" % (brain_mri, lh_pial, rh_pial))

    def check_anatomy_freeview(self):
        '''Opens Freeview with the orig.mgz MRI loaded along with the rCT and aparc.a2009s+aseg.mgz. 
        User should scroll through to check that the anatomical labels assigned to depths are correct.'''
        brain_mri = os.path.join(self.subj_dir, self.subj, 'mri', 'orig.mgz')
        elecs_CT = os.path.join(self.subj_dir, self.subj, 'CT', 'rCT.nii')
        aparc_aseg = os.path.join(self.subj_dir, self.subj, 'mri', 'aparc.a2009s+aseg.mgz')
        os.system("freeview --volume %s:opacity=0.8 --volume %s:opacity=0.6 --volume %s:colormap=lut:opacity=0.5:visible=0 --viewport 'coronal'"%(brain_mri, elecs_CT, aparc_aseg))
        
    def make_dural_surf(self, radius=3, num_iter=30, dilate=0.0):
        '''
        Create smoothed dural surface for projecting electrodes to.

        Parameters
        ----------  
        radius : float 
            radius for smoothing (currently ignored)
        num_iter : int
            Number of iterations for mris_smooth
        dilate : float
            Amount of dilation for dural surface (argument to mris_expand)

        Returns
        ----------
        None
        
        '''
        
        from surface_warping_scripts.make_outer_surf import make_outer_surf # From ielu
        # Create mask of pial surface
        hems = ['lh', 'rh']
        for hem in hems:
            print("Creating mask of %s pial surface"%(hem))
            pial_surf = os.path.join(self.subj_dir, self.subj, 'surf', hem+'.pial')
            pial_fill_image = os.path.join(self.subj_dir, self.subj, 'surf', hem+'.pial.filled.mgz')
            if not os.path.isfile(pial_fill_image):
                os.system('mris_fill -c -r 1 %s %s'%(pial_surf, pial_fill_image))
            outfile = os.path.join(self.subj_dir, self.subj, 'surf', hem+'.pial.outer')

            if os.path.isfile(pial_fill_image):
                # Create outer surface of this pial surface
                print("Creating outer surface for the filled pial surface %s"%(pial_fill_image))
                make_outer_surf(pial_surf, pial_fill_image, radius, outfile) # this is from ielu

                os.system('mris_extract_main_component %s %s-main'%(outfile, outfile))
                dura_surf = os.path.join(self.subj_dir, self.subj, 'surf', hem+'.dural')
                os.system('mris_smooth -nw -n %d %s-main %s'%(num_iter, outfile, dura_surf))

                if dilate != 0:
                    print("Dilating surface by %d mm"%(dilate))
                    if dilate > 0:
                        print("Multiplying dilate value by -1 to get outward dilation")
                        dilate = -1*dilate
                    os.system('mris_expand %s %d %s'%(dura_surf, dilate, dura_surf))
            else:
                print("Failed to create %s, check inputs."%(pial_fill_image))
        
        self.convert_fsmesh2mlab(mesh_name = 'dural')

    def mark_electrodes(self):
        ''' Launch the electrode picker for this subject. The electrode
        picker requires the Qt4Agg backend, so is launched via an external
        python script. 

        Inputs to the electrode_picker.py script include the subject directory and the hemisphere
        of implantation.
        '''
        individual_elecs_dir = os.path.join(self.subj_dir,self.subj,'elecs','individual_elecs')
        if not os.path.isdir(individual_elecs_dir):
            print("Creating directory %s"%(individual_elecs_dir))
            os.mkdir(individual_elecs_dir)
        print("Launching electrode picker")
        epicker = os.path.join(self.img_pipe_dir, 'SupplementalScripts', 'electrode_picker.py')
        os.system('python %s %s %s'%(epicker, os.path.join(self.subj_dir, self.subj), self.hem))

    def convert_fsmesh2mlab(self, mesh_name='pial'):
        '''Creates surface mesh triangle and vertex .mat files
        If no argument for mesh_name is given, lh.pial and rh.pial
        are converted into lh_pial_trivert.mat and rh_pial_trivert.mat
        in the Meshes directory (for use in python) and *_lh_pial.mat
        and *_rh_pial.mat for use in MATLAB.

        Parameters
        ----------
        mesh_name : {'pial', 'white', 'inflated'}
        
        '''

        hems = ['lh', 'rh']

        if not os.path.isdir(self.mesh_dir):
            print('Making Meshes Directory')
            # Make the Meshes directory in subj_dir if it does not yet exist
            os.mkdir(self.mesh_dir)

        # Loop through hemispheres for this mesh, create one .mat file for each
        for h in hems:
            print("Making %s mesh"%(h))
            mesh_surf = os.path.join(self.surf_dir, h+'.'+mesh_name)
            vert, tri = nib.freesurfer.read_geometry(mesh_surf)
            out_file = os.path.join(self.mesh_dir, '%s_%s_trivert.mat'%(h, mesh_name))
            out_file_struct = os.path.join(self.mesh_dir, '%s_%s_%s.mat'%(self.subj, h, mesh_name))
            scipy.io.savemat(out_file, {'tri': tri, 'vert': vert})

            cortex = {'tri': tri+1, 'vert': vert}
            scipy.io.savemat(out_file_struct, {'cortex': cortex})

        if mesh_name=='pial':
            self.pial_surf_file = dict()
            self.pial_surf_file['lh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'lh_pial_trivert.mat')
            self.pial_surf_file['rh'] = os.path.join(self.subj_dir, self.subj, 'Meshes', 'rh_pial_trivert.mat')
        else:
            setattr(self, mesh_name+'_surf_file', out_file)
  
    def reg_img(self, source='CT.nii', target='orig.mgz', smooth=0., reg_type='rigid', interp='pv', xtol=0.0001, ftol=0.0001):
        '''Runs nmi coregistration between two images.
        Usually run as patient.reg_img() 
        You can also specify the source (usually a CT scan, assumed to be in $SUBJECTS_DIR/subj/CT)
        and the target (usually T1 MRI, assumed to be in $SUBJECTS_DIR/subj/mri)

        Parameters are arguments for nipy.algorithms.registration.histogram_registration.HistogramRegistration
        (for more information, see help for nipy.algorithms.registration.optimizer).

        Parameters
        ----------
        smooth : float
            a smoothing parameter in mm
        reg_type : {'rigid', 'affine'}
            Registration type
        interp : {'pv','tri'} 
            changes the interpolation method for resampling
        xtol : float
            tolerance parameter for function minimization
        ftol : float
            tolerance parameter for function minimization

            
        '''

        source_file = os.path.join(self.CT_dir, source)
        target_file = os.path.join(self.mri_dir, target)

        print("Computing registration from %s to %s"%(source_file, target_file))
        ctimg  = nipy.load_image(source_file)
        mriimg = nipy.load_image(target_file)

        ct_cmap = ctimg.coordmap  
        mri_cmap = mriimg.coordmap

        # Compute registration
        ct_to_mri_reg = nipy.algorithms.registration.histogram_registration.HistogramRegistration(ctimg, mriimg, similarity='nmi', smooth=smooth, interp=interp)
        aff = ct_to_mri_reg.optimize(reg_type).as_affine()   

        ct_to_mri = AffineTransform(ct_cmap.function_range, mri_cmap.function_range, aff)  
        reg_CT = nipy.algorithms.resample.resample(ctimg, mri_cmap, ct_to_mri.inverse(), mriimg.shape)    

        outfile = os.path.join(self.CT_dir, 'r'+source)
        print("Saving registered CT image as %s"%(outfile))
        nipy.save_image(reg_CT, outfile)

    def interp_grid(self, nrows=16, ncols=16, grid_basename='hd_grid'):
        '''Interpolates corners for an electrode grid
        given the four corners (in order, 1, 16, 241, 256), or for
        32 channel grid, 1, 8, 25, 32.
        
        Parameters
        ----------
        nrows : int
            Number of rows in the grid
        ncols : int
            Number of columns in the grid
        grid_basename : str
            The base name of the grid (e.g. 'hd_grid' if you have a corners file
            called hd_grid_corners.mat)
        
        '''

        nchans = nrows*ncols

        corner_file = os.path.join(self.elecs_dir, 'individual_elecs', grid_basename+'_corners.mat')
        corners = scipy.io.loadmat(corner_file)['elecmatrix']
        elecmatrix = np.zeros((nchans, 3))

        corner_nums = [0, 0+nrows-1, nchans-nrows, nchans-1]

        # Add the electrode coordinates for the corners
        for i in np.arange(4):
            elecmatrix[corner_nums[i],:] = corners[i,:]

        # Interpolate over one dimension (vertical columns from corner 1 to 2 and corner 3 to 4)
        # loop over x, y, and z coordinates
        for i in np.arange(3):
            elecmatrix[corner_nums[0]:corner_nums[1]+1,i] = np.linspace(elecmatrix[corner_nums[0],i], elecmatrix[corner_nums[1], i], nrows)
            elecmatrix[corner_nums[2]:corner_nums[3]+1,i] = np.linspace(elecmatrix[corner_nums[2],i], elecmatrix[corner_nums[3], i], nrows)

        grid = np.arange(nchans).reshape((nrows, ncols), order='F')

        # Now fill in the rows using the new data from the leftmost and rightmost columns
        for row in np.arange(nrows):
            for i in np.arange(3):
                elecmatrix[grid[row,:],i] = np.linspace(elecmatrix[row,i], elecmatrix[row+(grid[0,-1]-grid[0,0]),i], ncols)

        orig_file = os.path.join(self.elecs_dir, 'individual_elecs', '%s_orig.mat'%(grid_basename))
        scipy.io.savemat(orig_file, {'elecmatrix': elecmatrix} )

    def extrap_grid(self, point_list=[(1,1),(1,5),(8,1)], nrows=16, ncols=16, grid_basename='hd_grid'):
        ''' This is to both interpolate and extrapolate grid coordinates when all 4 corners are not visible
        on a photograph or navigation software. The coordinates of the first corner must be available as well
        another point in the first row, and another point in the first column at least.

        First point should be corner, second point should be on first column and third should be on first row.

        Saves the the hd_grid_orig.mat and hd_grid_corners.mat files so that the grid can be projected.

        point_list contains the grid indices of the electrodes in the input points.

        hd_grid_points.mat contains the coordinates of the 3 input points.

        '''

        nchans = nrows*ncols

        points_file = os.path.join(self.elecs_dir, 'individual_elecs', grid_basename+'_points.mat')
        points = scipy.io.loadmat(points_file)['elecmatrix']
        elecmatrix = np.zeros((nchans, 3))
        grid = np.arange(nchans).reshape((nrows, ncols), order='F')
        points_nums = [0, point_list[1][1]-1, ncols*point_list[2][0]-1]

        # Add the electrode coordinates for the measured points
        for i in np.arange(3):
            elecmatrix[points_nums[i], :] = points[i, :]

        stepcol = np.zeros((3,1))
        # Interpolate over one dimension (vertical columns from corner 1 to second point
        # loop over x, y, and z coordinates
        for i in np.arange(3):
            linspacecol= np.linspace(elecmatrix[points_nums[0], i], elecmatrix[points_nums[1], i], point_list[1][1], retstep=True)
            elecmatrix[points_nums[0]:points_nums[1]+1, i] = linspacecol[0]
            stepcol[i] = linspacecol[1]

        for x in range(point_list[1][1], nrows):
            for i in np.arange(3):
                elecmatrix[x, i]=elecmatrix[x-1, i] + stepcol[i]

        steprow = np.zeros((3, 1))
        #Now interpolate/extrapolate the first row of the grid
        for i in np.arange(3):
            linspacerow = np.linspace(elecmatrix[0,i], elecmatrix[points_nums[2], i], point_list[2][0], retstep=True)
            elecmatrix[grid[0,points_nums[0]:point_list[2][0]], i] = linspacerow[0]
            steprow[i] = linspacerow[1]

        for x in range(point_list[2][0]-1, ncols):
            for i in np.arange(3):
                elecmatrix[grid[0, x], i] = elecmatrix[grid[0, x-1], i] + steprow[i]

        # Now fill in the rows using the new data from the leftmost and top
        for row in range(1, nrows):
            for col in range(1, ncols):
                for i in np.arange(3):
                    elecmatrix[grid[row, col], i] = elecmatrix[grid[row - 1, col], i] + stepcol[i]

        grid_file = os.path.join(self.elecs_dir, 'individual_elecs', grid_basename + '_orig.mat')
        scipy.io.savemat(grid_file, {'elecmatrix': elecmatrix})

        #now prepare the corner file

        cornermatrix = np.zeros((4, 3))
        cornermatrix[0, :] = elecmatrix[0, :]
        cornermatrix[1, :] = elecmatrix[grid[nrows-1, 0], :]
        cornermatrix[2, :] = elecmatrix[grid[0, ncols-1], :]
        cornermatrix[3, :] = elecmatrix[grid[nrows-1, ncols-1], :]

        corner_file = os.path.join(self.elecs_dir, 'individual_elecs', grid_basename+'_corners.mat')
        scipy.io.savemat(corner_file, {'elecmatrix': cornermatrix})



    def project_electrodes(self, elecfile_prefix='hd_grid', use_mean_normal=True, \
                                 surf_type='dural', \
                                 num_iter=30, dilate=0.0, grid=True,
                                 convex_hull=True):
        '''
        Project electrodes to the brain's surface to correct for deformation.
        
        By default, projects the electrodes of a grid based on the mean normal vector of the four grid
        corner electrodes that were manually localized from the registered CT. Can also project strips
        and individual electrodes if grid=False (you will be prompted for a projection direction).
        
        Parameters
        ----------
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix 
        use_mean_normal : bool, optional
            whether to use mean normal vector (mean of the 4 normal vectors from the grids 
            corner electrodes) as the projection direction
        surf_type : {'dural','pial','OFC'}, optional
            Which surface to project to.  In the case of the orbitofrontal cortex (OFC),
            it often works better to create a separate ROI of those brain structures and
            project to that, rather than to the pial or dural surface
        num_iter : int, optional
            Number of iterations for dural surface creation
        dilate : float, optional
            Amount to dilate the dural surface
        grid : bool, optional
            Whether the electrodes to project are from a grid that was interpolated using interp_grid()
        convex_hull : bool, optional
            Whether to use the convex hull of the relevant surface or not.  Often
            this can result in a smoother looking projection rather than a "wavy"
            looking grid.

        Returns
        ---------
        None

        '''
        
        from .surface_warping_scripts.project_electrodes_anydirection import project_electrodes_anydirection

        print('Projection Params: \n\t Grid Name: %s.mat \n\t Use Mean Normal: %s \n\t \
               Surface Type: %s \n\t Number of Smoothing Iterations (if using dural): %d'\
                %(elecfile_prefix,use_mean_normal,surf_type,num_iter))
        if grid: 
            corners_file = os.path.join(self.elecs_dir, 'individual_elecs', elecfile_prefix+'_corners.mat')
            elec_corners = scipy.io.loadmat(corners_file)['elecmatrix']
            elecfile_name = elecfile_prefix +'_orig'
            if not os.path.isdir(os.path.join(self.elecs_dir, 'individual_elecs', 'preproc')):
                print('Making preproc directory')
                os.mkdir(os.path.join(self.elecs_dir, 'individual_elecs','preproc'))
        else:
            elecfile_name = elecfile_prefix

        dural_mesh = os.path.join(self.subj_dir, self.subj, 'Meshes', self.subj + '_' + self.hem + '_dural.mat')
        if surf_type=='dural' and not os.path.isfile(dural_mesh):
            print('Creating dural surface mesh, using %d smoothing iterations'%(num_iter))
            self.make_dural_surf(num_iter=num_iter, dilate=dilate)

        if use_mean_normal and grid:
            #the corners draw out a 'rectangle', each side is a grid vector
            grid_vectors = elec_corners[1] - elec_corners[0], \
                           elec_corners[3] - elec_corners[2], \
                           elec_corners[3] - elec_corners[1], \
                           elec_corners[2] - elec_corners[0]
            #we can get 4 normal vectors from the 4 grid vectors 
            normal_vectors = [np.cross(grid_vectors[0],grid_vectors[2]), \
                              np.cross(grid_vectors[0],grid_vectors[3]), \
                              np.cross(grid_vectors[1],grid_vectors[2]), \
                              np.cross(grid_vectors[1],grid_vectors[3])]
            
            #normalize the normal vectors
            for vec in range(4):
                normal_vectors[vec] = normal_vectors[vec]/np.linalg.norm(normal_vectors[vec])

            #get all the normal vectors facing the same direction
            for vec in range(1,4):
                angle = np.arccos(np.dot(normal_vectors[vec],normal_vectors[vec-1])/(np.linalg.norm(normal_vectors[vec])*np.linalg.norm(normal_vectors[vec-1])))
                #flip the normal vector and see if angle between it and the previous normal vector is smaller
                flipped_angle = np.arccos(np.dot((-1.0*normal_vectors[vec]),normal_vectors[vec-1])/(np.linalg.norm(normal_vectors[vec])*np.linalg.norm(normal_vectors[vec-1])))
                if angle > flipped_angle:
                    normal_vectors[vec] = normal_vectors[vec]*-1.0
            print('Normal vectors: ' + str(normal_vectors))

            #take the mean of the 4 normal vectors
            mean_normal = np.array(normal_vectors).mean(axis=0)

            #mean normal vector should point outwards (laterally, not towards the center of the brain)
            #if it doesn't, flip the mean normal vector
            if (self.hem=='lh' and mean_normal[0] < 0) or (self.hem=='rh' and mean_normal[0] > 0):
                direction = [mean_normal[0],mean_normal[1],mean_normal[2]]
            else:
                direction = [-1.0*mean_normal[0],-1.0*mean_normal[1],-1.0*mean_normal[2]]

            # project the electrodes to the convex hull of the pial surface
            print('Projection direction vector: ', direction)
        else:
            proj_direction = raw_input('Enter a custom projection direction as a string (lh,rh,top,bottom,front,back,or custom): If none provided, will default to hemisphere: \n')
            if proj_direction == 'custom':
                x = raw_input('Enter projection vector\'s x-component: \n')
                y = raw_input('Enter projection vector\'s y-component: \n')
                z = raw_input('Enter projection vector\'s z-component: \n')
                direction = [float(x),float(y),float(z)]
            elif proj_direction not in ['lh','rh','top','bottom','front','back','custom']:
                direction = self.hem
            else:
                direction = proj_direction

        if elecfile_prefix == 'OFC_grid':
            self.make_roi_mesh('OFC', ['lateralorbitofrontal','medialorbitofrontal','rostralmiddlefrontal','parsorbitalis',
                                       'parstriangularis','superiorfrontal','rostralanteriorcingulate',
                                       'caudalanteriorcingulate','frontalpole','insula'], hem=None, showfig=False)
            surf_type = 'OFC'

        print('::: Loading Mesh data :::')
        print(os.path.join(self.subj_dir, self.subj, 'Meshes', '%s_%s_trivert.mat'%(self.hem, surf_type)))
        cortex = scipy.io.loadmat(os.path.join(self.subj_dir, self.subj, 'Meshes', '%s_%s_trivert.mat'%(self.hem, surf_type)))
        tri, vert = cortex['tri'], cortex['vert']

        if convex_hull:
            tri = scipy.spatial.Delaunay(vert).convex_hull

        print('::: Projecting electrodes to mesh :::')
        elecmatrix = scipy.io.loadmat(os.path.join(self.subj_dir, self.subj, 'elecs', 'individual_elecs', '%s_orig.mat'%(elecfile_prefix)))['elecmatrix']
        print(direction)
        elecs_proj = project_electrodes_anydirection(tri, vert, elecmatrix, direction)
        scipy.io.savemat(os.path.join(self.subj_dir, self.subj, 'elecs', 'individual_elecs', '%s.mat'%(elecfile_prefix)),{'elecmatrix':elecs_proj})
        print('::: Done :::')

        #move files to preproc subfolder
        if grid:
            corner_file = os.path.join(self.elecs_dir, 'individual_elecs', elecfile_prefix+'_corners.mat')
            orig_file = os.path.join(self.elecs_dir, 'individual_elecs', elecfile_prefix+'_orig.mat')
            preproc_dir = os.path.join(self.elecs_dir, 'individual_elecs', 'preproc') 
            print('Moving %s to %s'%(orig_file, preproc_dir))
            os.system('mv %s %s'%(corner_file, preproc_dir))
            os.system('mv %s %s'%(orig_file, preproc_dir))
        return 

    def get_clinical_grid(self, elecfile_prefix='hd_grid'):
        '''Loads in HD grid coordinates and use a list
        of indices to extract downsampled grid.
        
        Parameters
        ----------
        elecfile_prefix : str
            prefix of the .mat file with the electrode coordinates matrix

        Returns
        -------
        clinicalgrid : array-like
            n x 3 elecmatrix (every other row and every other column)
        '''

        # load in clinical grid indices
        clingrid = scipy.io.loadmat(os.path.join(self.img_pipe_dir, 'SupplementalFiles','clingrid_inds.mat'))['inds'].ravel()

        # load in high density grid coordinates
        hd = scipy.io.loadmat(os.path.join(self.elecs_dir, 'individual_elecs', elecfile_prefix+'.mat'))['elecmatrix']

        # If we have a grid that is smaller than 256 channels, remove the irrelevant clinical grid indices
        clingrid = clingrid[clingrid<hd.shape[0]]

        # get clinical grid coordinates using the relevant indices
        clinicalgrid = hd[clingrid,:]

        # save new coordinates
        scipy.io.savemat(os.path.join(self.elecs_dir, 'individual_elecs', 'clinical_'+elecfile_prefix+'.mat'), {'elecmatrix': clinicalgrid})

        return clinicalgrid

    def get_clinical_elecs_all(self,elecfile_prefix='TDT_elecs_all',num_grids=1,grid_shortnames=['G']):
        '''Loads in elecs all file created with high density grid
        and outputs elecs all file containing downsampled grid and no NaNs
        '''


        print('This only is to be used when the only difference is the density of grids and/or presence of NaNs')
        origelecs = scipy.io.loadmat(os.path.join(self.elecs_dir, elecfile_prefix+'.mat'))['elecmatrix']
        origlabels = scipy.io.loadmat(os.path.join(self.elecs_dir, elecfile_prefix+'.mat'))['anatomy']

        short_label = []
        long_label = []
        grid_or_depth = []
        hd_label = []
        masterlist = []

        for r in origlabels:
                short_label.append(r[0][0]) # This is the shortened electrode montage label
                long_label.append(r[1][0])  # This is the long form electrode montage label
                grid_or_depth.append(r[2][0])  # This is the label for grid, depth, or strip

        for string in range(0, len(short_label)):
            lead = []
            for s in short_label[string]:
                if not s.isdigit():
                    lead.append(s)
            masterlist.append(''.join(lead))

        clinicalgrids=[]
        for grid in range(0,num_grids):
            hd = np.empty((0, 3))
            coord = np.empty((1, 3))
            for string in range(0, len(short_label)):
                if grid_shortnames[grid] == masterlist[string]:
                    coord[0,:] = origelecs[string]
                    hd = np.append(hd, coord, axis=0)
                    hd_label.append(''.join(lead))

            # load in clinical grid indices
            clingrid = scipy.io.loadmat(os.path.join(self.img_pipe_dir, 'SupplementalFiles', 'clingrid_inds.mat'))['inds'].ravel()
            # If we have a grid that is smaller than 256 channels, remove the irrelevant clinical grid indices
            clingrid = clingrid[clingrid < hd.shape[0]]
            # get clinical grid coordinates using the relevant indices
            clinicalgrid = hd[clingrid, :]
            clinicalgrids.append(clinicalgrid)

        newelecs=np.empty((0,3))
        new_short_label=[]
        new_long_label=[]
        new_grid_depth=[]

        for string in range(0, len(short_label)):
            if masterlist[string] not in ('NaN', 'nan', 'ECG', 'EKG', 'NAN', 'EOG', 'ROC', 'LOC', 'EEG', 'EMG', 'scalpEEG'):
                if masterlist[string] not in grid_shortnames:
                    coord = np.empty((1, 3))
                    coord[0, :] = origelecs[string]
                    newelecs = np.append(newelecs, coord, axis=0)
                    new_short_label.append(short_label[string])
                    new_long_label.append(long_label[string])
                    new_grid_depth.append(grid_or_depth[string])

                else:  # electrode is part of a grid
                    for grid in range(0, num_grids):
                        if masterlist[string] == grid_shortnames[grid] and clinicalgrids[grid].shape[0] > 0:
                            clinicalgrid = clinicalgrids[grid]
                            coord = np.empty((1, 3))
                            coord[0, :] = clinicalgrid[0, :]
                            newelecs = np.append(newelecs, coord, axis=0)
                            clinicalgrids[grid] = np.delete(clinicalgrid, 0, axis=0)

                            new_short_label.append(short_label[string])
                            new_long_label.append(long_label[string])
                            new_grid_depth.append(grid_or_depth[string])


        newelecs = newelecs[newelecs != [0,0,0]]
        newelecmatrix = np.reshape(newelecs, (newelecs.shape[0]/3, 3))
        neweleclabels = np.empty((newelecmatrix.shape[0], 3), dtype=np.object)
        neweleclabels[:,0] = new_short_label
        neweleclabels[:,1] = new_long_label
        neweleclabels[:,2] = new_grid_depth

        scipy.io.savemat(os.path.join(self.elecs_dir, 'clinical_' + elecfile_prefix + '.mat'),
                         {'elecmatrix': newelecmatrix,'eleclabels': neweleclabels})


    def get_subcort(self):
        '''Obtains .mat files for vertex and triangle
           coords of all subcortical freesurfer segmented meshes'''

        # set ascii dir name
        subjAscii_dir = os.path.join(self.subj_dir, self.subj, 'ascii')
        if not os.path.isdir(subjAscii_dir):
            os.mkdir(subjAscii_dir)

        # tessellate all subjects freesurfer subcortical segmentations
        print('::: Tesselating freesurfer subcortical segmentations from aseg using aseg2srf... :::')
        print(os.path.join(self.img_pipe_dir, 'SupplementalScripts', 'aseg2srf.sh'))
        os.system(os.path.join(self.img_pipe_dir, 'SupplementalScripts', 'aseg2srf.sh') + ' -s "%s" -l "4 5 10 11 12 13 17 18 26 \
                 28 43 44  49 50 51 52 53 54 58 60 14 15 16" -d' % (self.subj))

        # get list of all .srf files and change fname to .asc
        srf_list = list(set([fname for fname in os.listdir(subjAscii_dir)]))
        asc_list = list(set([fname.replace('.srf', '.asc') for fname in srf_list]))
        asc_list.sort()
        for fname in srf_list:
            new_fname = fname.replace('.srf', '.asc')
            os.system('mv %s %s'%(os.path.join(subjAscii_dir,fname), os.path.join(subjAscii_dir,new_fname)))

        # convert all ascii subcortical meshes to matlab vert, tri coords
        subcort_list = ['aseg_058.asc', 'aseg_054.asc', 'aseg_050.asc',
                        'aseg_052.asc', 'aseg_053.asc', 'aseg_051.asc', 'aseg_049.asc',
                        'aseg_043.asc', 'aseg_044.asc', 'aseg_060.asc', 'aseg_004.asc',
                        'aseg_005.asc', 'aseg_010.asc', 'aseg_011.asc', 'aseg_012.asc',
                        'aseg_013.asc', 'aseg_017.asc', 'aseg_018.asc', 'aseg_026.asc',
                        'aseg_028.asc', 'aseg_014.asc', 'aseg_015.asc', 'aseg_016.asc']

        nuc_list = ['rAcumb', 'rAmgd', 'rCaud', 'rGP', 'rHipp', 'rPut', 'rThal',
                    'rLatVent', 'rInfLatVent', 'rVentDienceph', 'lLatVent', 'lInfLatVent',
                    'lThal', 'lCaud', 'lPut',  'lGP', 'lHipp', 'lAmgd', 'lAcumb', 'lVentDienceph',
                    'lThirdVent', 'lFourthVent', 'lBrainStem']

        subcort_dir = os.path.join(self.mesh_dir,'subcortical')     
        if not os.path.isdir(subcort_dir):      
            print('Creating directory %s'%(subcort_dir))        
            os.mkdir(subcort_dir)

        print('::: Converting all ascii segmentations to matlab tri-vert :::')
        for i in range(len(subcort_list)):
            subcort = os.path.join(subjAscii_dir, subcort_list[i])
            nuc = os.path.join(subcort_dir, nuc_list[i])
            self.subcortFs2mlab(subcort, nuc)

    def subcortFs2mlab(self, subcort, nuc):
        '''Function to convert freesurfer ascii subcort segmentations
           to triangular mesh array .mat style. This helper function is
           called by get_subcort().
        
        Parameters
        ----------
        subcort : str
            Name of the subcortical mesh ascii file (e.g. aseg_058.asc).
            See get_subcort().
        nuc : str
            Name of the subcortical nucleus (e.g. 'rAcumb')
        
        '''

        # use freesurfer mris_convert to get ascii subcortical surface
        subcort_ascii = subcort

        # clean up ascii file and extract matrix dimensions from header
        subcort = open(subcort_ascii, 'r')
        subcort_mat = subcort.readlines()
        subcort.close()
        subcort_mat.pop(0)  # get rid of comments in header
        # get rid of new line char
        subcort_mat = [item.strip('\n') for item in subcort_mat]

        # extract inds for vert and tri
        subcort_inds = subcort_mat.pop(0)
        # seperate inds into two strings
        subcort_inds = subcort_inds.split(' ')
        subcort_inds = [int(i) for i in subcort_inds]  # convert string to ints

        # get rows for vertices only, strip 0 column, and split into seperate
        # strings
        subcort_vert = [item.strip(' 0')
                        for item in subcort_mat[:subcort_inds[0]]]
        subcort_vert = [item.split('  ')
                        for item in subcort_vert]  # seperate strings

        # Convert into an array of floats
        subcort_vert = np.array(np.vstack((subcort_vert)), dtype=np.float)

        # get rows for triangles only, strip 0 column, and split into seperate
        # strings
        subcort_tri = [item[:-2] for item in subcort_mat[subcort_inds[0] + 1:]]
        subcort_tri = [item.split(' ')
                       for item in subcort_tri]  # seperate strings
        subcort_tri = np.array(np.vstack((subcort_tri)), dtype=np.int)

        outfile = '%s_subcort_trivert.mat' % (nuc)
        scipy.io.savemat(outfile, {'tri': subcort_tri, 'vert': subcort_vert})  # save tri/vert matrix

        # convert inds to scipy mat
        subcort_inds = scipy.mat(subcort_inds)
        scipy.io.savemat('%s_subcort_inds.mat' %
                         (nuc), {'inds': subcort_inds})  # save inds
        
        out_file_struct = '%s_subcort.mat' % (nuc)
        
        cortex = {'tri': subcort_tri+1, 'vert': subcort_vert}
        scipy.io.savemat(out_file_struct, {'cortex': cortex})

    def make_elecs_all(self, input_list=None, outfile=None):
        '''Interactively creates a .mat file with the montage and coordinates of 
        all the elecs files in the /elecs_individual folder.
        
        Parameters
        ----------
        input_list : list of tuples
            default is None which leads to the interactive prompts to create the elecs_all file, but you can
            input your own list of devices as a list of tuples, where each tuple is in the format
            (short_name, long_name, elec_type, filename). The filename is the elecmatrix .mat file of the 
            device, and should be in the elecs/individual_elecs folder. If you want to add NaN rows, 
            enter ('Nan','Nan','Nan',number_of_nan_rows), where in the fourth element you specify how many empty rows
            you'd like to add. 
        outfile : str
            the name of the file you want to save to, specify this if not using the interactive version

        Usage
        -----
        >>> patient.make_elecs_all(input_list=[('AD','AmygDepth','depth','amygdala_depth.mat'),('G','Grid','grid','hd_grid.mat'),('nan','nan','nan',5)], outfile='test_elecs_all_list')
        
        
        '''
        short_names,long_names, elec_types, elecmatrix_all = [],[],[], []

        #non-interactive version, with input_list and outfile specified
        if input_list != None:
            for device in input_list:
                short_name_prefix, long_name_prefix, elec_type, file_name = device[0], device[1], device[2], device[3]
                if short_name_prefix.lower() == 'nan' or long_name_prefix.lower() == 'nan' or elec_type.lower() == 'nan' or file_name == 'nan':
                    num_empty_rows = file_name
                    elecmatrix_all.append(np.ones((num_empty_rows,3))*np.nan)
                    short_names.extend([short_name_prefix for i in range(num_empty_rows)])
                    long_names.extend([long_name_prefix for i in range(num_empty_rows)])
                    elec_types.extend([elec_type for i in range(num_empty_rows)])
                else:
                    indiv_file = os.path.join(self.elecs_dir,'individual_elecs', file_name)
                    elecmatrix = scipy.io.loadmat(indiv_file)['elecmatrix']
                    num_elecs = elecmatrix.shape[0]
                    elecmatrix_all.append(elecmatrix)

                    short_names.extend([short_name_prefix+str(i) for i in range(1, num_elecs+1)])
                    long_names.extend([long_name_prefix+str(i) for i in range(1, num_elecs+1)])
                    elec_types.extend([elec_type for i in range(num_elecs)])
        else: #interactive
            done = False
            while done == False:
                num_empty_rows = raw_input('Are you adding a row that will be NaN in the elecmatrix? If not, press enter. If so, enter the number of empty rows to add: \n')
                if len(num_empty_rows):
                    num_empty_rows = int(num_empty_rows)
                    short_name_prefix = raw_input('What is the short name prefix (e.g. G, AD, HD)?\n')
                    short_names.extend([short_name_prefix for i in range(1,num_empty_rows+1)])
                    long_name_prefix = raw_input('What is the long name prefix (e.g. L256Grid, HippocampalDepth)?\n')
                    long_names.extend([long_name_prefix for i in range(1,num_empty_rows+1)])
                    elec_type = raw_input('What is the type of the device (e.g. grid, strip, depth)?\n')
                    elec_types.extend([elec_type for i in range(num_empty_rows)])
                    elecmatrix_all.append(np.ones((num_empty_rows,3))*np.nan)
                else:
                    short_name_prefix = raw_input('What is the short name prefix of the device (e.g. G, AD, HD)?\n')
                    long_name_prefix = raw_input('What is the long name prefix of the device? (e.g. L256Grid, HippocampalDepth)\n')
                    elec_type = raw_input('What is the type of the device? (e.g. grid, strip, depth)\n')     
                    try:
                        file_name = raw_input('What is the filename of the device\'s electrode coordinate matrix?\n')
                        indiv_file = os.path.join(self.elecs_dir,'individual_elecs', file_name)
                        elecmatrix = scipy.io.loadmat(indiv_file)['elecmatrix']
                    except IOError:
                        file_name = raw_input('Sorry, that file was not found. Please enter the correct filename of the device: \n ')
                        indiv_file = os.path.join(self.elecs_dir,'individual_elecs', file_name)
                        elecmatrix = scipy.io.loadmat(indiv_file)['elecmatrix']
                    num_elecs = elecmatrix.shape[0]
                    elecmatrix_all.append(elecmatrix)
                    short_names.extend([short_name_prefix+str(i) for i in range(1,num_elecs+1)])
                    long_names.extend([long_name_prefix+str(i) for i in range(1,num_elecs+1)])
                    elec_types.extend([elec_type for i in range(num_elecs)])
                completed = raw_input('Finished entering devices? Enter \'y\' if finished.')
                if completed=='y':
                    done = True 
            outfile = raw_input('What filename would you like to save out to?\n')
        elecmatrix_all = np.vstack(elecmatrix_all)
        eleclabels = np.ones(elecmatrix_all.shape,dtype=np.object)
        eleclabels[:,0] = short_names
        eleclabels[:,1] = long_names
        eleclabels[:,2] = elec_types
        label_outfile = os.path.join(self.elecs_dir, '%s.mat'%(outfile))
        scipy.io.savemat(label_outfile,{'eleclabels':eleclabels,'elecmatrix':elecmatrix_all})

    def edit_elecs_all(self, revision_dict, elecfile_prefix='TDT_elecs_all'):
        '''Edit the anatomy matrix of the elecfile_prefix. 
        
        Parameters
        ----------
        revision_dict : dict
            In each entry of the revision_dict, the key is the anatomical label 
            you'd like to impose on the value, which is a list of 0-indexed electrode numbers.
            For example, revision_dict = {'superiortemporal':[3,4,5],'precentral':[23,25,36]}
            would change electrodes 3, 4, and 5 to superiortemporal and 23, 25, and 36 to 
            precentral.
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix

        Returns
        -------
        elecs_all : dict
            Dictionary with keys 'elecmatrix' (list of coordinates) and 'anatomy'
            (anatomical labels for each electrode)

        '''
        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        elecs_all = scipy.io.loadmat(elecfile)
        anatomy = elecs_all['anatomy']
        for k,v in revision_dict.items():
            if self.zero_indexed_electrodes is False:
                v = np.array(v)+1
            for elec_num in v:
                anatomy[elec_num,3] = k
        elecs_all['anatomy'] = anatomy
        scipy.io.savemat(elecfile, elecs_all)

        return elecs_all

    def nearest_electrode_vert(self, cortex_verts, elecmatrix):
        ''' Find the vertex on a mesh that is closest to the given electrode
        coordinates.
        
        Parameters
        ----------
        cortex_verts : array-like
            [nvertices x 3] matrix of vertices on the cortical surface mesh
        elecmatrix : array-like
            [nchans x 3] matrix of 3D electrode coordinates 

        Returns
        -------
        vert_inds : array-like
            Array of vertex indices that are closest to each of the 
            electrode 
        nearest_verts : array-like
            Coordinates for the nearest cortical vertices
        '''

        nchans = elecmatrix.shape[0]
        d = np.zeros((nchans, cortex_verts.shape[0]))

        # Find the distance between each electrode and all possible vertices
        # on the surface mesh
        for chan in np.arange(nchans):
            d[chan,:] = np.sqrt((elecmatrix[chan,0] - cortex_verts[:,0])**2 + \
                                (elecmatrix[chan,1] - cortex_verts[:,1])**2 + \
                                (elecmatrix[chan,2] - cortex_verts[:,2])**2)

        # Find the index of the vertex nearest to each electrode
        vert_inds = np.argmin(d, axis = 1)
        nearest_verts = cortex_verts[vert_inds,:]

        return vert_inds, nearest_verts

    def label_elecs(self, elecfile_prefix='TDT_elecs_all', atlas_surf='desikan-killiany', atlas_depth='destrieux', elecs_all=True):
        ''' Automatically labels electrodes based on the freesurfer annotation file.
        Assumes TDT_elecs_all.mat or clinical_elecs_all.mat files
        Uses both the Desikan-Killiany Atlas and the Destrieux Atlas, as described 
        here: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
        
        Parameters
        ----------
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix
        atlas_surf : {'desikan-killiany', 'destrieux'}
            The atlas to use for labeling of surface electrodes.
        atlas_depth : {'destrieux', 'desikan-killiany'}
            The atlas to use for labeling of depth electrodes.
        elecs_all : bool
            Label all electrodes
        
        Returns
        -------
        elec_labels : array-like
            [nchans x 4] matrix of electrode labels. Columns include short name,
            long name, 'grid'/'depth'/'strip' label, and assigned anatomical label.
        '''

        if atlas_surf == 'desikan-killiany':
            surf_atlas_flag = ''
        elif atlas_surf == 'destrieux':
            surf_atlas_flag = '--a2009s'
        else:
            surf_atlas_flag = ''

        print(self.subj_dir)
        print('Creating labels from the freesurfer annotation file for use in automated electrode labeling')
        gyri_labels_dir = os.path.join(self.subj_dir, self.subj, 'label', 'gyri')
        if not os.path.isdir(gyri_labels_dir):
            os.mkdir(gyri_labels_dir)
         
        # This version of mri_annotation2label uses the coarse labels from the Desikan-Killiany Atlas, unless
        # atlas_surf is 'destrieux', in which case the more detailed labels are used
        os.system('mri_annotation2label --subject %s --hemi %s --surface pial %s --outdir %s'%(self.subj, self.hem, surf_atlas_flag, gyri_labels_dir))
        print('Loading electrode matrix')
        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        elecmatrix = scipy.io.loadmat(elecfile)['elecmatrix']
        
        # Initialize empty variable for indices of grid and strip electrodes
        isnotdepth = []
        
        # Choose only the surface or grid electrodes (if not using hd_grid.mat)
        if elecfile_prefix == 'TDT_elecs_all' or elecfile_prefix == 'clinical_elecs_all' or elecs_all:
            elecmontage = scipy.io.loadmat(elecfile)['eleclabels']
            # Make the cell array into something more usable by python
            short_label = []
            long_label = []
            grid_or_depth = []
 
            for r in elecmontage:
                short_label.append(r[0][0]) # This is the shortened electrode montage label
                long_label.append(r[1][0]) # This is the long form electrode montage label
                grid_or_depth.append(r[2][0]) # This is the label for grid, depth, or strip
            
            # These are the indices that won't be used for labeling
            #dont_label = ['EOG','ECG','ROC','LOC','EEG','EKG','NaN','EMG','scalpEEG']
            indices = [i for i, x in enumerate(long_label) if ('EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x==np.nan or 'scalpEEG' in x)]
            indices.extend([i for i, x in enumerate(short_label) if ('EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x==np.nan or 'scalpEEG' in x)])
            indices.extend([i for i, x in enumerate(grid_or_depth) if ('EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x==np.nan or 'scalpEEG' in x)])
            indices.extend(np.where(np.isnan(elecmatrix)==True)[0])
            indices = list(set(indices))
            indices_to_use = list(set(range(len(long_label))) - set(indices))

            # Initialize the cell array that we'll store electrode labels in later
            elec_labels_orig = np.empty((len(long_label),4),dtype=np.object)
            elec_labels_orig[:,0] = short_label
            elec_labels_orig[:,1] = long_label
            elec_labels_orig[:,2] = grid_or_depth 
            elec_labels = np.empty((len(indices_to_use),4), dtype = np.object)
            elecmatrix_orig = elecmatrix
            elecmatrix = elecmatrix[indices_to_use,:]
            
            short_label_orig,long_label_orig,grid_or_depth_orig = short_label,long_label,grid_or_depth
            short_label = [i for j, i in enumerate(short_label) if j not in indices]
            long_label = [i for j, i in enumerate(long_label) if j not in indices]
            grid_or_depth = [i for j, i in enumerate(grid_or_depth) if j not in indices]
            elec_labels[:,0] = short_label
            elec_labels[:,1] = long_label
            elec_labels[:,2] = grid_or_depth
            
            # Find the non depth electrodes
            isnotdepth = np.array([r!='depth' for r in grid_or_depth])
            
        # Use the surface label files to get which label goes with each surface vertex
        label_files = glob.glob(os.path.join(gyri_labels_dir, '%s.*.label'%(self.hem)))
        vert_label = {}
        for label in label_files:
            label_name = label.split('.')[1]
            print('Loading label %s'%label_name)
            fid = open(label,'r')
            d = np.genfromtxt(fid, delimiter=' ', skip_header=2)
            vertnum, x, y, z, junk=d[~np.isnan(d)].reshape((-1,5)).T
            for v in vertnum:
                vert_label[np.int(v)] = label_name.strip()
            fid.close()

        trivert_file = os.path.join(self.mesh_dir, '%s_pial_trivert.mat'%(self.hem))
        cortex_verts = scipy.io.loadmat(trivert_file)['vert']

        # Only use electrodes that are grid or strips
        if len(isnotdepth)>0:
            elecmatrix_new = elecmatrix[isnotdepth,:]
        else:
            elecmatrix_new = elecmatrix

        print('Finding nearest mesh vertex for each electrode')
        vert_inds, nearest_verts = self.nearest_electrode_vert(cortex_verts, elecmatrix_new)

        ## Now make a dictionary of the label for each electrode
        elec_labels_notdepth=[]
        for v in range(len(vert_inds)):
            if vert_inds[v] in vert_label:
                elec_labels_notdepth.append(vert_label[vert_inds[v]].strip())
            else:
                elec_labels_notdepth.append('Unknown')

        if elecfile_prefix == 'TDT_elecs_all' or elecfile_prefix == 'clinical_elecs_all' or elecs_all:
            elec_labels[isnotdepth,3] = elec_labels_notdepth
            elec_labels[np.invert(isnotdepth),3] = '' # Set these to an empty string instead of None type
        else:
            elec_labels = np.array(elec_labels_notdepth, dtype = np.object)
        print('Saving electrode labels for surface electrodes to %s'%(elecfile_prefix))
        ## added by BKD so that elec_mat_grid='hd_grid' works. It does not contain elecmontage
        save_dict = {'elecmatrix': elecmatrix, 'anatomy': elec_labels}
        if 'elecmontage' in locals():
            save_dict['eleclabels'] = elecmontage
        else:
            print('electmontage does not exist')
        #scipy.io.savemat('%s/%s/elecs/%s'%(self.subj_dir, self.subj, elecfile_prefix), save_dict)

        if np.any(np.invert(isnotdepth)): # If there are depth electrodes, run this part
            print('*************************************************')
            print('Now doing the depth electrodes')

            # Get the volume corresponding to the labels from the Destrieux atlas, which is more 
            # detailed than Desikan-Killiany (https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation)
            if atlas_depth == 'desikan-killiany':
                depth_atlas_nm = ''
            elif atlas_depth == 'destrieux':
                depth_atlas_nm = '.a2009s'
            else:
                depth_atlas_nm = '.a2009s'

            aseg_file = os.path.join(self.subj_dir, self.subj, 'mri', 'aparc%s+aseg.mgz'%(depth_atlas_nm))
            dat = nib.freesurfer.load(aseg_file)
            aparc_dat = dat.get_data()
             
            # Define the affine transform to go from surface coordinates to volume coordinates (as CRS, which is
            # the slice *number* as x,y,z in the 3D volume. That is, if there are 256 x 256 x 256 voxels, the
            # CRS coordinate will go from 0 to 255.)
            affine = np.array([[  -1.,    0.,    0.,  128.],
                               [   0.,    0.,    1., -128.],
                               [   0.,   -1.,    0.,  128.],
                               [   0.,    0.,    0.,    1.]])

            elecs_depths = elecmatrix[np.invert(isnotdepth),:]
            intercept = np.ones(len(elecs_depths))
            elecs_ones = np.column_stack((elecs_depths,intercept))

            # Find voxel CRS
            VoxCRS = np.dot(np.linalg.inv(affine), elecs_ones.transpose()).transpose().astype(int)
            # Make meshgrid the same size as aparc_dat (only for gaussian blob version), ignore
            #xx, yy, zz = np.mgrid[0:aparc_dat.shape[0], 0:aparc_dat.shape[1], 0:aparc_dat.shape[2]]
            #unique_labels = np.unique(aparc_dat)
            #unique_labels = unique_labels[unique_labels>0]

            # Get the names of these labels using Freesurfer's lookup table (LUT)
            print("Loading lookup table for freesurfer labels")
            fid = open(os.path.join(self.fs_dir,'FreeSurferColorLUT.txt'))
            LUT = fid.readlines()
            fid.close()

            # Make dictionary of labels
            LUT = [row.split() for row in LUT]
            lab = {}
            for row in LUT:
                if len(row)>1 and row[0][0] is not '#' and row[0][0] is not '\\': # Get rid of the comments
                    lname = row[1]
                    lab[np.int(row[0])] = lname

            # Label the electrodes according to the aseg volume
            nchans = VoxCRS.shape[0]
            anatomy = np.empty((nchans,), dtype=np.object)
            print("Labeling electrodes...")

            for elec in np.arange(nchans):
                anatomy[elec] = lab[aparc_dat[VoxCRS[elec,0], VoxCRS[elec,1], VoxCRS[elec,2]]]
                print("E%d, Vox CRS: [%d, %d, %d], Label #%d = %s"%(elec, VoxCRS[elec,0], VoxCRS[elec,1], VoxCRS[elec,2], 
                                                                    aparc_dat[VoxCRS[elec,0], VoxCRS[elec,1], VoxCRS[elec,2]], 
                                                                    anatomy[elec]))

            elec_labels[np.invert(isnotdepth),3] = anatomy
            
            #make some corrections b/c of NaNs in elecmatrix
        elec_labels_orig[:,3] = ''
        elec_labels_orig[indices_to_use,3] = elec_labels[:,3] 
        
        print('Saving electrode labels to %s'%(elecfile_prefix))
        scipy.io.savemat(os.path.join(self.elecs_dir, elecfile_prefix+'.mat'), {'elecmatrix': elecmatrix_orig, 
                                                                                'anatomy': elec_labels_orig, 
                                                                                'eleclabels': elecmontage})

        return elec_labels

    def warp_all(self, elecfile_prefix='TDT_elecs_all', warp_depths=True, warp_surface=True, template='cvs_avg35_inMNI152'):
        ''' Warps surface and depth electrodes and runs quality checking functions for them. 

        Parameters
        ----------
        elecfile_prefix : str, optional
            the name of the .mat file with the electrode coordinates in elecmatrix
        warp_depths : bool, optional
            whether to warp depth electrodes
        warp_surface : bool, optional
            whether to warp surface electrodes 
        template : str, optional
            which atlas brain to use (default: 'cvs_avg35_inMNI152').  Must be an atlas
            in the freesurfer subjects directory.

        Returns
        -------
        Creates the file depthWarpsQC.pdf in patient.elecs_dir, which can be used for depth warping
        quality checking. 
        '''

        print("Using %s as the template for warps"%(template))

        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        elecfile_warped = os.path.join(self.elecs_dir, '%s_warped.mat'%(elecfile_prefix))
        elecfile_nearest_warped = os.path.join(self.elecs_dir, '%s_nearest_warped.mat'%(elecfile_prefix))
        elecfile_nearest_warped_text = os.path.join(self.elecs_dir, elecfile_prefix+'_nearest_warped.txt')
        elecfile_RAS_text = os.path.join(self.elecs_dir, elecfile_prefix+'_RAS.txt')
        
        if os.path.isfile(elecfile_warped):
            print("The electrodes in %s have already been warped and are in %s"%(elecfile, elecfile_warped))
            return
        
        orig_elecs = scipy.io.loadmat(elecfile)

        if 'depth' in orig_elecs['anatomy'][:,2] and warp_depths:
            #if mri_cvs_register already run, don't run again
            if not os.path.isfile(os.path.join(self.subj_dir, self.subj, 'cvs', 'combined_to'+template+'_elreg_afteraseg-norm.tm3d')):
                self.get_cvsWarp(template)
            else:
                print('%s registration file already created, proceeding to apply the depth warp'%(os.path.join(self.subj_dir, self.subj, 'cvs', 'combined_to'+template+'_elreg_afteraseg-norm.tm3d')))
            if not os.path.isfile(elecfile_nearest_warped):
                self.apply_cvsWarp(elecfile_prefix,template)
            else:
                print("Depth warping has already been applied to the depth electrodes of %s and are in %s"\
                    %(elecfile, elecfile_nearest_warped))

            depth_warps = scipy.io.loadmat(elecfile_nearest_warped)
            depth_indices = np.where(orig_elecs['anatomy'][:,2]=='depth')[0]
            if depth_warps['elecmatrix'].size > 0:
                orig_elecs['elecmatrix'][depth_indices] = depth_warps['elecmatrix']
            if not os.path.isfile(os.path.join(self.elecs_dir,'depthWarpsQC.pdf')):
                self.check_depth_warps(elecfile_prefix,template)
        
        if warp_surface:
            #if surface warp already run, don't run again
            if not os.path.isfile(os.path.join(self.subj_dir,self.subj,'elecs', elecfile_prefix + '_surface_warped.mat')):
                self.get_surface_warp(elecfile_prefix,template)
            else:
                print('Found %s, not running surface warp again'%(os.path.join(self.subj_dir,self.subj,'elecs', elecfile_prefix + '_surface_warped.mat')))
            elecfile_surface_warped = os.path.join(self.elecs_dir, elecfile_prefix+'_surface_warped.mat')
            surface_warps = scipy.io.loadmat(elecfile_surface_warped)
            surface_indices = np.array(list(set(np.where(orig_elecs['anatomy'][:,2]!='depth')[0]) & set(np.where(np.all(~np.isnan(orig_elecs['elecmatrix']),axis=1))[0])),dtype='int64')
            if surface_warps['elecmatrix'].size > 0:
                orig_elecs['elecmatrix'][surface_indices,:] = surface_warps['elecmatrix']

        #if both depth and surface warping have been done, create the combined warp .mat file
        if warp_depths and warp_surface:
            scipy.io.savemat(elecfile_warped,{'elecmatrix':orig_elecs['elecmatrix'],'anatomy':orig_elecs['anatomy']})
            
            self.plot_recon_anatomy_compare_warped(elecfile_prefix=elecfile_prefix)

            if not os.path.isdir(os.path.join(self.elecs_dir, 'warps_preproc')):
                print('Making preproc directory')
                os.mkdir(os.path.join(self.elecs_dir, 'warps_preproc'))
            preproc_dir = os.path.join(self.elecs_dir, 'warps_preproc')
            if os.path.isfile(elecfile_surface_warped):
                os.system('mv %s %s;' %(elecfile_surface_warped, preproc_dir))
            if os.path.isfile(elecfile_nearest_warped):
                os.system('mv %s %s'%(elecfile_nearest_warped, preproc_dir))
            if os.path.isfile(elecfile_nearest_warped_text):
                os.system('mv %s %s'%(elecfile_nearest_warped_text, preproc_dir))
            if os.path.isfile(elecfile_RAS_text):
                os.system('mv %s %s'%(elecfile_RAS_text, preproc_dir))

    def get_cvsWarp(self, template='cvs_avg35_inMNI152', openmp_threads=4):
        '''Method for obtaining nonlinearly warped MNI coordinates using 
        freesurfer's mri_cvs_register
        
        Parameters
        ----------
        template : str, optional
            Template brain for nonlinear warp (default is 'cvs_avg35_inMNI152')
        openmp_threads : int, optional
            Number of openmp threads for parallelization of mri_cvs_register

        Returns
        -------
        None.

        '''

        # run cvs register
        orig = self.subj  # orig is mri in fs orig space

        print('::: Computing Non-linear warping from patient native T1 to fs CVS MNI152 :::')
        os.system('mri_cvs_register --mov %s --template %s --nocleanup --openmp %d' % (orig, template, openmp_threads))
        print('cvsWarp COMPUTED')

    def apply_cvsWarp(self, elecfile_prefix='TDT_elecs_all',template_brain='cvs_avg35_inMNI152'):
        ''' Apply the CVS warp from mri_cvs_register to the electrode file of your choice. 
        
        Parameters
        ----------
        elecfile_prefix : str, optional
            the name of the .mat file with the electrode coordinates in elecmatrix
        template_brain : str, optional
            Template brain for nonlinear warp (default is 'cvs_avg35_inMNI152')
    
        Returns
        -------
        elecmatrix : array-like
            [nchans x 3] electrode matrix after nonlinear warping
        '''

        elecmatrix = np.empty((0, 4), int)

        fsVox2RAS = np.array(
            [[-1., 0., 0., 128.], [0., 0., 1., -128.], [0., -1., 0., 128.], [0., 0., 0., 1.]])

        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        elecs = scipy.io.loadmat(elecfile)
        anatomy = elecs.get('anatomy')
        depth_indices = np.where(anatomy[:,2]=='depth')[0]
        elec = elecs.get('elecmatrix')[depth_indices,:]

        #### WARP ALL ELECS AT ONCE ####
        print(':::: Computing RAS2Vox and saving %s txt file ::::'%elecfile_prefix)
        if len(elec.shape) > 1:
           intercept = np.ones(len(elec));
           elec = np.column_stack((elec,intercept));
        else:
           elec = np.concatenate((elec, np.ones(1)), axis=0)

        # Convert surface RAS to voxel CRS
        VoxCRS = np.dot(np.linalg.inv(fsVox2RAS), elec.transpose()).transpose()
        orig_file = os.path.join(self.mri_dir, 'orig.mgz')
        affine_file = os.path.join(self.mri_dir, 'affine_subj.txt')
        os.system('mri_info --vox2ras %s > %s'%(orig_file, affine_file))
        #affine_subj = np.loadtxt(affine_file)

        #affine_template = np.loadtxt(os.path.join(self.subj_dir, '%s_affine_subj.txt'%(template_brain)))

        elec = VoxCRS
        elec = elec[:,0:3]
        nan_elecs = np.isnan(elec)
        elec = np.nan_to_num(elec)

        RAS_text_file = os.path.join(self.elecs_dir, elecfile_prefix+'_RAS.txt')
        np.savetxt(RAS_text_file, elec, fmt='%1.2f\t%1.2f\t%1.2f', delimiter='\t')

        print(':::: Applying Non-linear warping ::::')
        template_brain_file = os.path.join(self.subj_dir, template_brain, 'mri', 'brain.mgz')
        morph_file = os.path.join(self.subj_dir, self.subj, 'cvs', 'combined_to'+template_brain+'_elreg_afteraseg-norm.tm3d')
        nearest_warped_file = os.path.join(self.elecs_dir, elecfile_prefix+'_nearest_warped.txt')
        os.system('applyMorph --template %s --transform %s tract_point_list %s %s nearest' % (template_brain_file,
                                                                                              morph_file,
                                                                                              RAS_text_file,
                                                                                              nearest_warped_file))

        print(':::: Computing Vox2RAS and saving %s_warped mat file ::::'% (elecfile_prefix))
        elec = np.loadtxt(nearest_warped_file)

        elec = np.concatenate((elec, np.ones((elec.shape[0], 1))), axis = 1)

        elecmatrix = np.dot(fsVox2RAS, elec.transpose()).transpose()  

        elecmatrix = elecmatrix[:, 0:3]
        
        # Set the electrodes back to NaN where applicable
        elecmatrix[np.where(nan_elecs[:,0]),:] = np.nan

         # This is for deleting the duplicate row that applyMorph produces for some reason
        if (elecmatrix[-1,:] == elecmatrix[-2,:]).all():
            elecmatrix = elecmatrix[:-1,:]
        nearest_warped_matfile = os.path.join(self.elecs_dir, elecfile_prefix+'_nearest_warped.mat')
        scipy.io.savemat(nearest_warped_matfile, {'elecmatrix': elecmatrix, 'anatomy': anatomy})

        return elecmatrix

    # Method to perform surface warps
    def get_surface_warp(self, basename='TDT_elecs_all', template='cvs_avg35_inMNI152'):
        ''' Perform surface warps on [basename].mat file, warping to template [template]
        which should also be present in the freesurfer $SUBJECTS_DIR
        
        Parameters
        ----------
        basename : str, optional
            prefix of the .mat file with the electrode coordinates matrix 
        template : str, optional
            Name of the template atlas for performing the surface warp 
            (default: 'cvs_avg35_inMNI152')

        '''               
        
        elecfile = os.path.join(self.elecs_dir,'%s_surface_warped.mat'%(basename))

        if os.path.isfile(elecfile):
            print("Surface warp file exists")
        else:
            print("Computing surface warp")

            elecmatrix = scipy.io.loadmat(os.path.join(self.elecs_dir, basename + '.mat'))['elecmatrix']
            anatomy = scipy.io.loadmat(os.path.join(self.elecs_dir, basename + '.mat'))['anatomy']
            labelpath = os.path.join(self.subj_dir, self.subj, 'label')
            
            if anatomy is not None:
                surface_indices = np.array(
                list(set(np.where(anatomy[:, 2] != 'depth')[0]) & set(np.where(np.all(~np.isnan(elecmatrix), axis=1))[0])),
                dtype='int64')
                anatomy = anatomy[surface_indices]
                elecmatrix = elecmatrix[surface_indices,:]
            
            elecs_warped = self.compute_surface_warp(elecmatrix, anatomy, labelpath, basename, template)

            scipy.io.savemat(elecfile, {'elecmatrix': np.array(elecs_warped), 'anatomy': anatomy})

            print("Surface warp for %s complete. Warped coordinates in %s" % (self.subj, elecfile))


    def compute_surface_warp(self, elecmatrix, anatomy=None, labelpath='tmp/', basename='',
                             template='cvs_avg35_inMNI152'):
        """
        wrapper that uses freesurfer's label_mri2mri to warp electrodes from one cortical surface to another.
        Parameters
        ----------
        elecmatrix : array-like
          n x 3 np.array
        anatomy : array-like
          np.array from TDT_elecs_all
        labelpath : str
          can be os.path.join(self.subj_dir, self.subj, 'label') or any other directory if you do not
          have write permissions there. Default is 'tmp/', which is deleted when function finishes.
        basename : str
          default = ''
        template : str
          default='cvs_avg35_inMNI152'
        
        Returns
        -------
        warped_elecs : array-like
          (n x 3 np.array)
        """

        labels_to_warp_path = os.path.join(labelpath, 'labels_to_warp')
        warped_labels_dir = os.path.join(labelpath, 'warped_labels')

        if not os.path.isdir(labelpath):
            os.mkdir(labelpath)
        if not os.path.isdir(warped_labels_dir):
            os.mkdir(warped_labels_dir)
        if not os.path.isdir(labels_to_warp_path):
            os.mkdir(labels_to_warp_path)

        cortex_src = self.get_surf()
        atlas_file = os.path.join(self.subj_dir, template, 'Meshes', self.hem + '_pial_trivert.mat')
        if not os.path.isfile(atlas_file):
            atlas_patient = freeCoG(subj=template, subj_dir=self.subj_dir, hem=self.hem)
            print("Creating mesh %s" % (atlas_file))
            atlas_patient.convert_fsmesh2mlab()

        if anatomy is not None:
            surface_indices = np.array(
                list(set(np.where(anatomy[:, 2] != 'depth')[0]) & set(np.where(np.all(~np.isnan(elecmatrix), axis=1))[0])),
                dtype='int64')
        else:
            surface_indices = range(len(elecmatrix))

        print("Finding nearest surface vertex for each electrode")
        vert_inds, nearest_verts = self.nearest_electrode_vert(cortex_src['vert'], elecmatrix[surface_indices, :])
        elecmatrix = nearest_verts

        print('Warping each electrode separately:')
        elecs_warped = np.nan * np.ones((len(surface_indices), 3))

        for c in np.arange(len(surface_indices)):
            chan = surface_indices[c]
            # Open label file for writing
            labelname_nopath = '%s.%s.chan%03d.label' % (self.hem, basename, chan)
            labelname = os.path.join(labels_to_warp_path, labelname_nopath)

            with open(labelname, 'w') as fid:
                fid.write('%s\n' % (labelname))
                # Print header of label file
                fid.write('#!ascii label  , from subject %s vox2ras=TkReg\n1\n' % (self.subj))
                fid.write('%i %.9f %.9f %.9f 0.0000000' % (vert_inds[chan], elecmatrix[chan, 0],
                                                           elecmatrix[chan, 1], elecmatrix[chan, 2]))

            print("Warping ch %d" % (chan))
            trglabel = os.path.join(warped_labels_dir, '%s.to.%s.%s' % (self.subj, template, labelname_nopath))
            os.system('mri_label2label --srclabel ' + labelname + ' --srcsubject ' + self.subj + \
                      ' --trgsubject ' + template + ' --trglabel ' + trglabel + ' --regmethod surface --hemi ' + self.hem + \
                      ' --trgsurf pial --paint 6 pial --sd ' + self.subj_dir)

            # Get the electrode coordinate from the label file
            with open(trglabel, 'r') as fid2:
                coord = fid2.readlines()[2].split()  # Get the third line

            elecs_warped[c, :] = ([np.float(coord[1]), np.float(coord[2]), np.float(coord[3])])
            # else:
            #     print("Channel %d is a depth electrode, not warping"%(chan))
            #     elecs_warped.append([np.nan, np.nan, np.nan])

            # intersect, t, u, v, xcoor = TriangleRayIntersection(elec, [1000, 0, 0], vert1,vert2,vert3, fullReturn=True)

        if labelpath == 'tmp/':
            shutil.rmtree(labels_to_warp_path)
            shutil.rmtree(warped_labels_dir)

        return elecs_warped


    def check_depth_warps(self, elecfile_prefix='TDT_elecs_all',template='cvs_avg35_inMNI152',atlas_depth='destrieux'):
        ''' Function to check whether warping of depths in mri_cvs_register worked properly. 
        Generates a pdf file with one page for each depth electrode, showing that electrode
        in the original surface space as well as in warped CVS space.  
        
        Parameters
        ----------
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix 
        template : str
            Name of the atlas template to use
        atlas_depth : {'destrieux', 'desikan-killiany'}
            Which atlas to use for depth labeling. Destrieux is more detailed 
            so is usually a good choice for depth electrodes.

        Returns
        -------
        Creates depthWarpsQC.pdf in patient.elecs_dir, which can be used to do quality
        checking on the depth warping.

        '''
        #get all subj elecs
        RAS_file = os.path.join(self.elecs_dir, '%s_RAS.txt'%(elecfile_prefix))
        subj_elecs,subj_elecnums = self.get_depth_elecs(RAS_file,'\n','\t',elecfile_prefix)

        nearest_warped_file = os.path.join(self.elecs_dir, '%s_nearest_warped.txt'%(elecfile_prefix))
        warped_elecs,warped_elecnums = self.get_depth_elecs(nearest_warped_file,'\n',' ',elecfile_prefix)

        #template brain (cvs)
        if atlas_depth == 'desikan-killiany':
            depth_atlas_nm = ''
        elif atlas_depth == 'destrieux':
            depth_atlas_nm = '.a2009s'
        else:
            depth_atlas_nm = '.a2009s'

        #template brain (cvs)
        cvs_img=nib.freesurfer.load(os.path.join(self.subj_dir, template, 'mri', 'aparc' + depth_atlas_nm + '+aseg.mgz'))
        cvs_dat=cvs_img.get_data()

        #subj brain 
        subj_img=nib.freesurfer.load(os.path.join(self.mri_dir, 'aparc' + depth_atlas_nm + '+aseg.mgz'))
        subj_dat=subj_img.get_data()

        pdf = PdfPages(os.path.join(self.elecs_dir, 'depthWarpsQC.pdf'))
        for i in range(len(subj_elecnums)): 
            if subj_elecs[i][0] != 0 and subj_elecs[i][0] != 10000:
                self.plot_elec(subj_elecs[i], warped_elecs[i], subj_dat, cvs_dat, subj_elecnums[i], pdf)
        pdf.close()

    def apply_xfm(self, xfm_dir='mri/transforms', xfm_file='talairach.xfm', 
                  source_file='elecs/TDT_elecs_all.mat', target_file='elecs/TDT_elecs_all_2tal.mat', 
                  file_type='elecs'):
        ''' Apply an xfm transform from freesurfer
        Parameters
        ----------
        xfm_dir : str
            Directory containing the xfm transform file.  Assumed to be a subdirectory
            within $SUBJECTS_DIR/[subj_id].  Default: 'mri/transforms'
        xfm_file : str
            Name of the xfm file.  Default: 'talairach.xfm'
        source_file : str
            Name of the file you want to apply the transform to.  Default: 'elecs/TDT_elecs_all.mat'
            If transforming a surface, use Meshes/[hem]_pial_trivert.mat
        target_file : str
            Name of the output file that has been transformed according to the transform
            in [xfm_file]. Default: 'elecs/TDT_elecs_all_2tal.mat'.  
            If transforming a surface, use e.g. 'Meshes/[hem]_pial_trivert_2tal.mat'
        file_type : {'surf', 'elecs'}
            The type of file to which you are applying the transform. Choices include
            'surf' (for a triangle-mesh surface) or 'elecs' (for an electrode matrix file,
            assumed to be in the elecs_all format).
        
        '''

        # First load the matrix in the xfm file:
        xfm_mat = np.genfromtxt(os.path.join(self.patient_dir, xfm_dir, xfm_file), skip_header=5, 
                                delimiter=' ', comments=';')

        # Also get the vox2RAS matrices for volumes and surfaces
        orig_file = os.path.join(self.mri_dir, 'orig.mgz')
        vox2ras_file = os.path.join(self.mri_dir, 'vox2ras.txt')
        vox2rastk_file = os.path.join(self.mri_dir, 'vox2rastk.txt')
        os.system('mri_info --vox2ras %s > %s'%(orig_file, vox2ras_file))
        os.system('mri_info --vox2ras-tkr %s > %s'%(orig_file, vox2rastk_file))
        
        vox2ras = np.loadtxt(vox2ras_file)
        vox2rastk = np.loadtxt(vox2rastk_file)

        print("Applying xfm %s to %s"%(xfm_file, source_file))
        if file_type == 'elecs':
            e = scipy.io.loadmat(os.path.join(self.patient_dir, source_file))
            elecs_ones = np.hstack((e['elecmatrix'], np.ones((e['elecmatrix'].shape[0],1)) ))
            elecs_xfm = reduce(np.dot, [xfm_mat, vox2ras, np.linalg.inv(vox2rastk), elecs_ones.T]).T
            print("Saving transformed coordinates as %s"%(target_file))
            scipy.io.savemat(os.path.join(self.patient_dir, target_file), 
                             {'elecmatrix': elecs_xfm, 'anatomy': e['anatomy']})

        elif file_type == 'surf':
            s = scipy.io.loadmat(os.path.join(self.patient_dir, source_file))
            vert_ones = np.hstack((s['vert'], np.ones((s['vert'].shape[0],1)) ))
            vert_xfm = reduce(np.dot, [xfm_mat, vox2ras, np.linalg.inv(vox2rastk), vert_ones.T]).T
            print("Saving transformed coordinates as %s"%(target_file))
            scipy.io.savemat(os.path.join(self.patient_dir, target_file), 
                             {'tri': s['tri'], 'vert': vert_xfm})

        else:
            warning("Please specify file_type = 'surf' or file_type = 'elecs'.")

    
    def apply_transform(self, elecfile_prefix, reorient_file):
        ''' Apply an affine transform (from SPM) to an electrode file.  

        Parameters
        ----------
        elecfile_prefix : str
            prefix of the .mat file with the electrode coordinates matrix 
        reorient_file : str
            Name of the mat file containing an SPM reorient matrix.

        Example
        -------
             patient.apply_transform(elecfile_prefix = 'TDT_elecs_all', reorient_file = 'T1_reorient')
        assumes transform is located in the subject's acpc directory
        '''
        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        elecs = scipy.io.loadmat(elecfile)
        elec = elecs.get('elecmatrix')
        anatomy = elecs.get('anatomy')
        
        reorient_file =  os.path.join(self.subj_dir, self.subj, 'acpc', reorient_file+'.mat')
        rmat = scipy.io.loadmat(reorient_file)['M']
        elecs_reoriented = nib.affines.apply_affine(rmat, elec)
        
        print("Saving reoriented electrodes")
        reoriented_elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'_reoriented.mat')
        scipy.io.savemat(reoriented_elecfile, {'elecmatrix': elecs_reoriented, 'anatomy': anatomy})
        print("Done.")

     #helper method to check the cvs depth warps:
    def plot_elec(self, orig_coords, warped_coords, subj_dat, cvs_dat, elec_num,pdf):
        ''' helper method to check the cvs depth warps. Each electrode is one page      
        in the resulting PDF.       
        Top row shows electrodes warped to the CVS brain, bottom row shows the electrodes       
        in their original position on the subject's brain. If the anatomical label      
        for the warped location matches that of the original subject brain, it is counted       
        as a "MATCH" and has a title in green, otherwise it is a "MISMATCH" and is      
        marked with a red title.        
        '''

        fs_lut = os.path.join(self.img_pipe_dir, 'SupplementalFiles', 'FreeSurferLUTRGBValues.npy')
        cmap = matplotlib.colors.ListedColormap(np.load(fs_lut)[:cvs_dat.max()+1,:])

        lookupTable = os.path.join(self.img_pipe_dir, 'SupplementalFiles', 'FreeSurferLookupTable')
        lookup_dict = pickle.load(open(lookupTable,'rb'))
        fig = plt.figure(figsize=((30,17)))
        nonzero_indices = np.where(cvs_dat>0)
        offset = 35 #this is how much you want to trim the mri by, there is a lot of empty space

        cvs_dat = cvs_dat[offset:-offset,offset:-offset,offset:-offset]
        cvs_vox_CRS = np.array([warped_coords[0]-offset,warped_coords[1]-offset,warped_coords[2]-offset],dtype='int')

        plt.subplot(2,3,1)
        plt.imshow(cvs_dat[cvs_vox_CRS[0],:,:], cmap=cmap)
        plt.plot(cvs_vox_CRS[2],cvs_vox_CRS[1],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()

        plt.subplot(2,3,2)
        plt.imshow(cvs_dat[:,cvs_vox_CRS[1],:].T, cmap=cmap)
        plt.plot(cvs_vox_CRS[0],cvs_vox_CRS[2],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()
        #t=plt.text(10,25,lookup_dict[cvs_dat[cvs_vox_CRS[0],cvs_vox_CRS[1],cvs_vox_CRS[2]]],color='white',size=25)
        plt.title('CVS brain ' + lookup_dict[cvs_dat[cvs_vox_CRS[0],cvs_vox_CRS[1],cvs_vox_CRS[2]]],size=20)

        plt.subplot(2,3,3)
        plt.imshow(cvs_dat[:,:,cvs_vox_CRS[2]].T, cmap=cmap)   
        plt.plot(cvs_vox_CRS[0],cvs_vox_CRS[1],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()

        subj_dat = subj_dat[offset:-offset,offset:-offset,offset:-offset]
        subj_vox_CRS = np.array([orig_coords[0]-offset,orig_coords[1]-offset,orig_coords[2]-offset],dtype='int')

        ax1=plt.subplot(2,3,4).axes
        plt.imshow(subj_dat[subj_vox_CRS[0],:,:], cmap=cmap)
        plt.plot(subj_vox_CRS[2],subj_vox_CRS[1],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()

        ax2=plt.subplot(2,3,5).axes
        plt.imshow(subj_dat[:,subj_vox_CRS[1],:].T, cmap=cmap)
        plt.plot(subj_vox_CRS[0],subj_vox_CRS[2],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()
        #t=plt.text(10,25,lookup_dict[subj_dat[subj_vox_CRS[0],subj_vox_CRS[1],subj_vox_CRS[2]]],color='white',size=25)
        plt.title(self.subj + ' ' +lookup_dict[subj_dat[subj_vox_CRS[0],subj_vox_CRS[1],subj_vox_CRS[2]]],size=20)

        ax3=plt.subplot(2,3,6).axes
        plt.imshow(subj_dat[:,:,subj_vox_CRS[2]].T, cmap=cmap)   
        plt.plot(subj_vox_CRS[0],subj_vox_CRS[1],'r*',markersize=14,color='#FFFFFF')
        plt.axis('tight'); ax = plt.gca(); ax.set_axis_off()

        if cvs_dat[cvs_vox_CRS[0],cvs_vox_CRS[1],cvs_vox_CRS[2]] != subj_dat[subj_vox_CRS[0],subj_vox_CRS[1],subj_vox_CRS[2]]:
            result = 'MISMATCH'
            color = 'r'
        else:
            result = 'Match'
            color = 'g'
        plt.suptitle('elec ' +str(elec_num) + ' '+result, size=30,color=color)
        pdf.savefig(fig)
        plt.close()

    def get_depth_elecs(self, fpath, delim1, delim2, elecfile_prefix):
        ''' Helper method to check cvs depth warps.  This helper method         
        just loads the electrode coordinates of the depth electrodes from the matlab file

        Parameters
        ----------
        fpath : str
            Path to the RAS coordinates file
        delim1 : str
            First delimiter (usually '\n')
        delim2 : str
            Second delimiter (usually '\t')
        elecfile_prefix : str
            prefix of the .mat file with the electrode coordinates matrix            
        '''
        f = open(fpath,'r')
        elecs = (f.read().split(delim1))
        elecs = np.array([e for e in elecs if len(e)>1])
        elecfile = os.path.join(self.elecs_dir, elecfile_prefix+'.mat')
        tdt_elec_types = scipy.io.loadmat(elecfile)['anatomy'][:,2][:-1]
        # depth_elecs = elecs[np.where(tdt_elec_types == 'depth')[0]]
        str_to_float = np.vectorize(lambda x: float(x))
        depth_elecs = str_to_float(np.array([s.split(delim2) for s in elecs]))
        return depth_elecs,np.where(tdt_elec_types == 'depth')[0]

    def get_elecs(self, elecfile_prefix='TDT_elecs_all', roi=None):
        '''Utility function to get electrode coordinate matrix of all electrodes in a certain anatomical region.

        Parameters
        ----------
        elecfile_prefix : str
            prefix of the .mat file with the electrode coordinates matrix 
        roi : str
            Which region of interest to load electrodes from.  See get_rois(). If roi=None,
            returns all the electrodes.

        Returns
        -------
        e : dict
            Dictionary containing 'elecmatrix' and 'anatomy'

        '''

        e = {'elecmatrix': [], 'anatomy': []}
        elecfile = os.path.join(self.elecs_dir,'%s.mat'%(elecfile_prefix))
        if os.path.isfile(elecfile):
            e = scipy.io.loadmat(elecfile)
            if roi != None:
                roi_indices = np.where(e['anatomy'][:,3]==roi)[0]
                elecmatrix = e['elecmatrix'][roi_indices,:]
                anatomy = e['anatomy'][roi_indices,:]
            else:
                elecfile = scipy.io.loadmat(os.path.join(self.elecs_dir,'%s.mat'%(elecfile_prefix)))
                return elecfile

            e = {'elecmatrix': elecmatrix, 'anatomy': anatomy} #{'anatomy': anatomy, 'elecmatrix': elecmatrix, 'eleclabels': eleclabels}
        else:
            print('File not found: %s'%(elecfile))
        return e

    def get_surf(self, hem='', roi='pial', template=None):
        ''' Utility for loading the pial surface for a given hemisphere ('lh' or 'rh') 
        
        Parameters
        ----------
        hem : {'', 'lh', 'rh'}
            Hemisphere for the surface. If blank, defaults to self.hem
        roi : str
            The region of interest to load.  Should be a Mesh that exists
            in the subject's Meshes directory, called [roi]_trivert.mat
        template : str, optional
            Name of the template to use if plotting electrodes on an atlas brain. e.g. 'cvs_avg35_inMNI152'
        
        Returns
        -------
        cortex : dict
            Dictionary containing 'tri' and 'vert' for the loaded region of interest mesh.

        '''
        if hem == '':
            hem = self.hem
        if roi == 'pial':
            if hem == 'lh' or hem == 'rh':
                if template == None:
                    cortex = scipy.io.loadmat(self.pial_surf_file[hem])
                else:
                    template_file = os.path.join(self.subj_dir, template, 'Meshes', hem+'_pial_trivert.mat')
                    cortex = scipy.io.loadmat(template_file)
            elif hem == 'stereo':
                cortex = dict()
                for h in ['lh', 'rh']:
                    if template == None:
                        cortex[h] = scipy.io.loadmat(self.pial_surf_file[h])
                    else:
                        template_file = os.path.join(self.subj_dir, template, 'Meshes', h+'_pial_trivert.mat')
                        cortex[h] = scipy.io.loadmat(template_file)
        else: 
            cortex = scipy.io.loadmat(os.path.join(self.mesh_dir, hem + '_' + roi + '_trivert.mat'))
        return cortex

    class roi:

        def __init__(self, name, color=(0.8,0.8,0.8), opacity=1.0, representation='surface', gaussian=False):
            '''Class defining a region of interest (ROI) mesh. This could be, for example, a mesh
            for the left hippocampus, or a mesh for the right superior temporal gyrus.

            Parameters
            ----------
            name : str
                name of the ROI
            color : tuple
                Tuple for the ROI's color where each value is between 0.0 and 1.0. 
            opacity : float (between 0.0 and 1.0)
                opacity of the mesh, between 0.0 and 1.0
            representation : {'surface', 'wireframe'}
                surface representation type
            gaussian: bool
                specifies how to represent electrodes and their weights on this mesh, note that setting gaussian to True 
                means the mesh color cannot be specified by the user and will instead use colors from the loaded
                colormap.

            Attributes
            ----------
            name : str
            color : tuple
            opacity : float (between 0.0 and 1.0)
            representation : {'surface', 'wireframe'}
            gaussian : bool 

            '''
                       
            self.name = name
            self.color = color
            self.opacity = opacity
            self.representation = representation
            self.gaussian = gaussian

    def get_rois(self):
        ''' Method to get the list of all available ROI names for plotting 
        
        Returns
        -------
        rois : list
            list of available regions of interest (ROIs) for plotting or making tri-vert meshes.

        '''
        rois = ['rAcumb', 'rAmgd', 'rCaud', 'rGP', 'rHipp', 'rPut', 'rThal',
                    'rLatVent', 'rInfLatVent', 'rVentDienceph', 'lLatVent', 'lInfLatVent',
                    'lThal', 'lCaud', 'lPut',  'lGP', 'lHipp', 'lAmgd', 'lAcumb', 'lVentDienceph',
                    'lThirdVent', 'lFourthVent', 'lBrainStem', 'pial', 'lh_pial', 'rh_pial', 'bankssts',
                    'inferiorparietal', 'medialorbitofrontal', 'pericalcarine', 'superiorfrontal',
                    'caudalanteriorcingulate', 'inferiortemporal', 'middletemporal','postcentral', 
                    'superiorparietal', 'caudalmiddlefrontal', 'insula', 'paracentral', 'posteriorcingulate',
                    'superiortemporal', 'cuneus', 'isthmuscingulate', 'parahippocampal', 'precentral',
                    'supramarginal', 'entorhinal', 'lateraloccipital', 'parsopercularis', 'precuneus',
                    'temporalpole', 'frontalpole', 'lateralorbitofrontal', 'parsorbitalis', 'rostralanteriorcingulate', 
                    'transversetemporal', 'fusiform', 'lingual', 'parstriangularis', 'rostralmiddlefrontal']
        return rois

    def run_annotation2label(self):
        ''' Create the labels from the Desikan-Killiany Atlas, which can be used for labeling or for 
        making ROIs. '''
        #create gyri labels directory
        gyri_labels_dir = os.path.join(self.subj_dir, self.subj, 'label', 'gyri')
        if not os.path.isdir(gyri_labels_dir):
            os.mkdir(gyri_labels_dir)
            # This version of mri_annotation2label uses the coarse labels from the Desikan-Killiany Atlas
            os.system('mri_annotation2label --subject %s --hemi %s --surface pial --outdir %s'\
                %(self.subj, 'lh', gyri_labels_dir))
            os.system('mri_annotation2label --subject %s --hemi %s --surface pial --outdir %s'\
                %(self.subj, 'rh', gyri_labels_dir))

    def plot_brain(self, rois=[roi(name='pial', color=(0.8,0.8,0.8), opacity=1.0, representation='surface', gaussian=False)], elecs=None, 
                    weights=None, cmap = 'RdBu', showfig=True, screenshot=False, helper_call=False, vmin=None, vmax=None,
                    azimuth=None, elevation=90, template=None):
        '''Plots multiple meshes on one figure. Defaults to plotting both hemispheres of the pial surface.
        
        Parameters
        ----------
        rois : list of roi objects 
            (create an roi object like so:
            hipp_roi = patient.roi(name='lHipp', color=(0.5,0.1,0.8), opacity=1.0, 
	                           representation='surface', gaussian=True)). 
            See get_rois() method for available ROI names.
        elecs : array-like
            [nchans x 3] electrode coordinate matrix
        weights : array-like 
            [nchans x 3] weight matrix associated with the electrode coordinate matrix
        cmap : str
            String specifying what colormap to use
        showfig : bool
            whether to show the figure in an interactive window
        screenshot: bool
            whether to save a screenshot and show using matplotlib (usually inline a notebook)
        helper_call : bool
            if plot_brain being used as a helper subcall, don't close the mlab instance
        vmin : float 
            Minimum color value when using cmap
        vmax : float
            Maximum color value when using cmap
        azimuth : float
            Azimuth for brain view.  By default (if azimuth=None), this will be set automatically to
            the left side for hem='lh', right side for hem='rh', or front view for 'pial'
        elevation : float
            Elevation for brain view. Default: 90
        template : None or str
            Name of the template to use if plotting electrodes on an atlas brain. e.g. 'cvs_avg35_inMNI152'
        
        Returns
        -------
        mesh : mayavi brain mesh
        points : mayavi electrodes
        mlab : mayavi mlab scene
        arr : array-like
            brain image screenshot
        fh : the mayavi figure handle

        Example
        -------
        >>> pial, hipp = patient.roi('pial',(0.6,0.3,0.6),0.1,'wireframe',True), patient.roi('lHipp',(0.5,0.1,0.8),1.0,'surface',True)
        >>> elecs = patient.get_elecs()['elecmatrix']
        >>> patient.plot_brain(rois=[pial,hipp],elecs=elecs,weights=np.random.uniform(0,1,(elecs.shape[0])))
        '''

        import mayavi
        from .plotting import ctmr_brain_plot as ctmr_brain_plot
        
        fh=mayavi.mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1200,900))

        #if there are any rois with gaussian set to True, don't plot any elecs as points3d, to avoid mixing gaussian representation
        #if needed, simply add the elecs by calling el_add
        any_gaussian = False

        for roi in rois:
            roi_name, color, opacity, representation, gaussian = roi.name, roi.color, roi.opacity, roi.representation, roi.gaussian

            #if color or opacity == None, then use default values 
            if color == None:
                color = (0.8, 0.8, 0.8)
            if opacity == None:
                opacity = 1.0
            if representation == None:
                representation = 'surface'
            if gaussian == None:
                gaussian = False
            if gaussian == True:
                any_gaussian = True

            # setup kwargs for ctmr_brain_plot.ctmr_gauss_plot()
            kwargs = {}
            kwargs.update(color=color, opacity=opacity, representation=representation, new_fig=False, cmap=cmap)
            if gaussian:
                kwargs.update(elecs=elecs, weights=weights, vmin=vmin, vmax=vmax)

            #default roi_name of 'pial' plots both hemispheres' pial surfaces
            if roi_name  in ('pial', 'rh_pial', 'lh_pial'):
                #use pial surface of the entire hemisphere
                if roi_name in ('pial', 'lh_pial'):
                    lh_pial = self.get_surf(hem='lh', template=template)
                    mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(lh_pial['tri'], lh_pial['vert'], **kwargs)

                if roi_name in ('pial', 'rh_pial'):
                    rh_pial = self.get_surf(hem='rh', template=template)
                    mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(rh_pial['tri'], rh_pial['vert'], **kwargs)
                    
            else:
                subcort_dir = os.path.join(self.mesh_dir,'subcortical')
                if os.path.isdir(subcort_dir) and '%s_subcort_trivert.mat'%(roi_name) in os.listdir(subcort_dir):
                    roi_mesh = scipy.io.loadmat(os.path.join(subcort_dir,'%s_subcort_trivert.mat'%(roi_name)))
                else:
                    roi_mesh = scipy.io.loadmat(os.path.join(self.mesh_dir,'%s_trivert.mat'%(roi_name)))

                mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(roi_mesh['tri'], roi_mesh['vert'], **kwargs)

        if not any_gaussian and elecs is not None:
            if weights is None: #if elecmatrix passed in but no weights specified, default to all ones for the electrode color weights
                points, mlab = ctmr_brain_plot.el_add(elecs)
            else:
                # Map the weights onto the current colormap
                elec_colors = cm.get_cmap(cmap)(weights)[:, :3]
                points, mlab = ctmr_brain_plot.el_add(elecs, color = elec_colors)

        else:
            #if no elecs to add as points3D
            points = None
        
        if azimuth is None:
            if self.hem == 'lh':
                azimuth = 180
            elif self.hem == 'rh':
                azimuth = 0
            else:
                azimuth = 90

        mlab.view(azimuth, elevation)

        if screenshot:
            arr = mlab.screenshot(antialiased=True)
            plt.figure(figsize=(20, 10))
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            plt.show()
        else:
            arr = []

        if showfig:
            mlab.show()

        if not helper_call and not showfig:
            mlab.close()

        return mesh, points, mlab, arr, fh
    
    def plot_recon_anatomy(self, elecfile_prefix='TDT_elecs_all', template=None, showfig=True, screenshot=False, 
                           opacity=1.0, show_numbers=True):
        '''
        Plot the brain along with all of the anatomically labeled electrodes, colored by location using freesurfer
        color lookup table.

        Parameters
        ----------
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix 
        template : str, optional
            Name of the template to use if plotting electrodes on an atlas brain
        showfig : bool
            Whether to show the figure or not
        screenshot : bool
            Whether to take a 2D screenshot or not
        opacity : float (from 0.0 to 1.0)
            opacity of the brain surface mesh.

        '''
        from .plotting import ctmr_brain_plot as ctmr_brain_plot
        from .SupplementalFiles import FS_colorLUT as FS_colorLUT

        a = self.get_surf(template=template)
        e = self.get_elecs(elecfile_prefix=elecfile_prefix)

        # Plot the pial surface
        if self.hem == 'lh' or self.hem == 'rh':
            mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(a['tri'], a['vert'], color=(0.8, 0.8, 0.8), opacity=opacity)
        elif self.hem == 'stereo':
            mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(a['lh']['tri'], a['lh']['vert'], color=(0.8, 0.8, 0.8), opacity=opacity)
            mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(a['rh']['tri'], a['rh']['vert'], color=(0.8, 0.8, 0.8), opacity=opacity, new_fig=False)

        # Add the electrodes, colored by anatomical region
        elec_colors = np.zeros((e['elecmatrix'].shape[0], e['elecmatrix'].shape[1]))

        # Import freesurfer color lookup table as a dictionary
        cmap = FS_colorLUT.get_lut()

        if self.hem=='lh':
            label_offset=-1.5
        elif self.hem=='rh':
            label_offset=1.5
        else:
            label_offset=1.5

        # Find all the unique brain areas in this subject
        brain_areas = np.unique(e['anatomy'][:,3])

        # Loop through unique brain areas and find the appropriate color for each brain area from the color LUT dictionary
        if self.hem=='stereo':
            label_hem = 'lh'
        else:
            label_hem = self.hem

        good_labels = []
        color_list = []
        for b in brain_areas:
            # Add relevant extra information to the label if needed for the color LUT
            if b[0][0] != 'NaN':
                this_label = b[0]
                good_labels.append(this_label)
                if b[0][0:3]!='ctx' and b[0][0:4] != 'Left' and b[0][0:5] != 'Right' and b[0][0:5] != 'Brain' and b[0] != 'Unknown':
                    this_label = 'ctx-%s-%s'%(label_hem, b[0])
                    print(this_label)
                
                if this_label != '' and this_label != 'NaN':
                    if this_label not in cmap:
                        #in case the label was manually assigned, and not found in the LUT colormap dictionary
                        el_color = matplotlib.cm.get_cmap('viridis').colors[int(float(np.where(brain_areas==b)[0])/float(len(brain_areas)))]
                    else:
                        el_color = np.array(cmap[this_label])/255.
                    color_list.append(el_color)
            elec_indices = np.where(e['anatomy'][:,3]==b)[0]
            elec_colors[elec_indices,:] = el_color

        if show_numbers:
            if self.zero_indexed_electrodes:
                elec_numbers = range(e['elecmatrix'].shape[0])
            else:
                elec_numbers = range(1,e['elecmatrix'].shape[0]+1)
        else:
            elec_numbers = None

        ctmr_brain_plot.el_add(e['elecmatrix'],elec_colors,numbers=elec_numbers, label_offset=label_offset)

        if self.hem=='lh':
            azimuth=180
        elif self.hem=='rh':
            azimuth=0
        else:
            azimuth=90

        mlab.view(azimuth, elevation=90)

        mlab.title('%s recon anatomy'%(self.subj),size=0.3)

        #arr = mlab.screenshot(antialiased=True)
        if screenshot:
            plt.figure(figsize=(20,20))
            arr, xoff, yoff = remove_whitespace(arr)
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            dummies = [plt.gca().plot([],[], ls='-', lw=5, c=c)[0] for c in color_list]
            plt.gca().legend(dummies, good_labels, ncol=4)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(self.patient_dir, '%s_anatomy.png'%self.subj))
        if showfig:
            mlab.show()
        #else:
        #    mlab.close()
        return mesh, mlab

    def plot_erps(self, erp_matrix, elecfile_prefix='TDT_elecs_all', time_scale_factor=0.03, 
                  z_scale_factor=3.0, showfig=True, screenshot=False, anat_colored=True):
        ''' Plot ERPs as traces on the brain surface.

        Parameters
        ----------
        erp_matrix : array like
        elecfile_prefix : str, optional
                          prefix of the .mat file with the electrode coordinates matrix 
        time_scale_factor : float
        z_scale_factor : float
        showfig : bool
        screenshot : bool
        anat_colored : bool

        '''
        from .SupplementalFiles import FS_colorLUT as FS_colorLUT
        #use mean normal vector
        #no anat color map option
        #time *-1 if left hemisphere
        #centering the time series more accurately
        #plotting all lines without the loop
        #mean normal as the offset for the erp lines' x-coordinate

        cmap = FS_colorLUT.get_lut()

        num_timepoints = erp_matrix.shape[1]
        num_channels = erp_matrix.shape[0]

        mesh, points, mlab, arr, f = self.plot_brain(showfig=False, helper_call=True)
        elecmatrix = self.get_elecs()['elecmatrix']
        if anat_colored:
            anatomy_labels = scipy.io.loadmat(os.path.join(self.elecs_dir, elecfile_prefix+'.mat'))['anatomy'][:,3]

        brain_areas = np.unique(anatomy_labels)
        for c in range(num_channels):
            if anat_colored:
                b = anatomy_labels[c]
                if b != 'NaN':
                    this_label = b[0]
                    if b[0][0:3]!='ctx' and b[0][0:4] != 'Left' and b[0][0:5] != 'Right' and b[0][0:5] != 'Brain' and b[0] != 'Unknown':
                        this_label = 'ctx-%s-%s'%('lh', b[0])

                    if this_label != '':
                        if this_label not in cmap:
                            #in case the label was manually assigned, and not found in the LUT colormap dictionary
                            erp_color = matplotlib.cm.get_cmap('viridis').colors[int(float(np.where(brain_areas==b)[0])/float(len(brain_areas)))]
                        else:
                            erp_color = np.array(cmap[this_label])/255.
            else:
                erp_color = (1., 0., 0.)
            erp_color = tuple(erp_color)
            elec_coord = elecmatrix[c,:]
            #mlab.points3d(elec_coord[0],elec_coord[1],elec_coord[2], scale_factor = 0.25, color = (1.0, 0.0, 0.0), resolution=25)
            erp = erp_matrix[c,:]

            if self.hem == 'lh':
                label_offset = -2.0
                y_array = ((np.array([i for i in range(num_timepoints/2-num_timepoints,num_timepoints/2)]))*time_scale_factor+elec_coord[1])[::-1]
            elif self.hem == 'rh':
                label_offset = 2.0
                y_array = ((np.array([i for i in range(num_timepoints/2-num_timepoints,num_timepoints/2)]))*time_scale_factor+elec_coord[1])

            x_array = np.array([elec_coord[0] for i in range(num_timepoints)])+label_offset
            
            z_array = erp*z_scale_factor+elec_coord[2]
            mlab.plot3d(x_array, y_array, z_array, figure=mlab.gcf(),color=(erp_color),tube_radius=0.15,tube_sides=3)
        
        mlab.view(distance=300)

        arr = mlab.screenshot(antialiased=True)
        if screenshot:
            plt.figure(figsize=(20,10))
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            plt.show()

        if showfig:
            mlab.show()
        mlab.close()
        return mesh, points, mlab

    def plot_recon_anatomy_compare_warped(self, template='cvs_avg35_inMNI152', elecfile_prefix='TDT_elecs_all',
                                          showfig=True, screenshot=False, opacity=1.0):
        ''' This plots two brains, one in native space, one in the template space, showing
        the native space and warped electrodes for ease of comparison/quality control.
        
        Parameters
        ----------
        template : str, optional
            which atlas brain to use (default: 'cvs_avg35_inMNI152').  Must be an atlas
            in the freesurfer subjects directory.
        elecfile_prefix : str, optional
            prefix of the .mat file with the electrode coordinates matrix 
        showfig : bool
            Whether to show the figure
        screenshot : bool
            Whether to take a 2D screenshot
        opacity : float (must be between 0.0 and 1.0)
            Opacity of surface meshes

        '''
        from .plotting import ctmr_brain_plot as ctmr_brain_plot
        from .SupplementalFiles import FS_colorLUT as FS_colorLUT

        subj_brain = self.get_surf()
        template_brain = self.get_surf(template=template)

        # Get native space and warped electrodes
        subj_e = self.get_elecs(elecfile_prefix=elecfile_prefix)
        template_e = self.get_elecs(elecfile_prefix=elecfile_prefix+'_warped')
        
        subj_brain_width = np.abs(np.max(subj_brain['vert'][:,1])-np.min(subj_brain['vert'][:,1]))
        template_brain_width = np.abs(np.max(template_brain['vert'][:,1])-np.min(template_brain['vert'][:,1]))
        
        subj_e['elecmatrix'][:,1] = subj_e['elecmatrix'][:,1]-((subj_brain_width+template_brain_width)/4)
        template_e['elecmatrix'][:,1] = template_e['elecmatrix'][:,1]+((subj_brain_width+template_brain_width)/4)

        subj_brain['vert'][:,1] = subj_brain['vert'][:,1]-((subj_brain_width+template_brain_width)/4)
        template_brain['vert'][:,1] = template_brain['vert'][:,1]+((subj_brain_width+template_brain_width)/4)
        
        # Plot the pial surface
        subj_mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(subj_brain['tri'], subj_brain['vert'], color=(0.8, 0.8, 0.8))
        template_mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(template_brain['tri'], template_brain['vert'], color=(0.8, 0.8, 0.8),new_fig=False)

        # Import freesurfer color lookup table as a dictionary
        cmap = FS_colorLUT.get_lut()

        # Make a list of electrode numbers
        if self.zero_indexed_electrodes:
            print("Using zero indexed electrodes")
            elec_numbers = np.arange(subj_e['elecmatrix'].shape[0])
        else:
            print("Using one indexed electrodes")
            elec_numbers = np.arange(subj_e['elecmatrix'].shape[0])+1

        # Find all the unique brain areas in this subject
        brain_areas = np.unique(subj_e['anatomy'][:,3])

        # Loop through unique brain areas and plot the electrodes in each brain area
        for b in brain_areas:
            # Add relevant extra information to the label if needed for the color LUT
            if b != 'NaN':
                if b[0][0:3]!='ctx' and b[0][0:4] != 'Left' and b[0][0:5] != 'Right' and b[0][0:5] != 'Brain' and b[0] != 'Unknown':
                    this_label = 'ctx-%s-%s'%(self.hem, b[0])
                    print(this_label)
                    if this_label != '':
                        if this_label not in cmap:
                            #in case the label was manually assigned, and not found in the LUT colormap dictionary
                            el_color = matplotlib.cm.get_cmap('viridis').colors[int(float(np.where(brain_areas==b)[0])/float(len(brain_areas)))]
                        else:
                            el_color = np.array(cmap[this_label])/255.
                    ctmr_brain_plot.el_add(np.atleast_2d(subj_e['elecmatrix'][subj_e['anatomy'][:,3]==b,:]), 
                                           color=tuple(el_color), numbers=elec_numbers[subj_e['anatomy'][:,3]==b])

                    ctmr_brain_plot.el_add(np.atleast_2d(template_e['elecmatrix'][subj_e['anatomy'][:,3]==b,:]), 
                                           color=tuple(el_color), numbers=elec_numbers[subj_e['anatomy'][:,3]==b])
        if self.hem == 'lh':
            azimuth = 180
        elif self.hem == 'rh':
            azimuth = 0
        mlab.view(azimuth, elevation=90)

        #adjust transparency of brain mesh
        subj_mesh.actor.property.opacity = opacity
        template_mesh.actor.property.opacity = opacity 

        if screenshot:
            plt.figure(figsize=(20,10))
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            plt.show()
            arr = mlab.screenshot(antialiased=True)
        if showfig:
            mlab.show()
        return subj_mesh, template_mesh, mlab

    def make_roi_mesh(self, roi_name, label_list, hem=None, showfig=False, save=True):

        ''' This function makes a mesh for the cortical ROI you are interested in. 
        
        Parameters
        ----------
        roi_name : str
            what you want to call your mesh, note that the hemisphere will be prepended to this name
        label_list : list 
             A list of labels, selected from the list below

            [bankssts               inferiorparietal        medialorbitofrontal     pericalcarine             superiorfrontal
            caudalanteriorcingulate inferiortemporal        middletemporal          postcentral               superiorparietal
            caudalmiddlefrontal     insula                  paracentral             posteriorcingulate        superiortemporal
            cuneus                  isthmuscingulate        parahippocampal         precentral                supramarginal
            entorhinal              lateraloccipital        parsopercularis         precuneus                 temporalpole
            frontalpole             lateralorbitofrontal    parsorbitalis           rostralanteriorcingulate  transversetemporal
            fusiform                lingual                 parstriangularis        rostralmiddlefrontal]
        
        hem : {None, 'lh', 'rh'}
            If None, defaults to the implantation hemisphere.  Otherwise uses the hemisphere specified by the user.
        showfig : bool
            Show figure or not.
        save : bool
            If save=True, this mesh is saved to the $SUBJECTS_DIR/Meshes/.  Otherwise it is not saved.

        Returns
        -------
        roi_mesh : dict
            A dictionary that contains roi_mesh['tri'] and roi_mesh['vert']
            and can be used in plotting commands for the region of interest triangle-vertex mesh.

        '''

        if hem==None:
            if self.hem != 'lh' and self.hem != 'rh':
                print('You need to specify which hemisphere this ROI is in. Please try again and specify the hemisphere in the hem argument.')
                return
            else:
                hem = self.hem

        cortex = self.get_surf(hem=hem)

        roi_mesh = {}
        vertnums = []

        for label in label_list:
            this_label = os.path.join(self.patient_dir, 'label', 'gyri', '%s.%s.label'%(hem, label))
            f = open(this_label, 'r')
            all_lines_str = [x.split() for x in f.read().split('\n')[2:][:-1]]
            all_lines_float = []
            for line in range(len(all_lines_str)):
                all_lines_float.append([float(x) for x in all_lines_str[line]])
            verts = np.array(all_lines_float, dtype=np.int)
            vertnums.extend(verts[:,0].tolist())
        vertnums = sorted(vertnums)
        roi_mesh['vert'] = cortex['vert'][vertnums, :]
        
        # Find the triangles for these vertex numbers
        tri_row_inds = np.sort(np.array(list(set(np.where(np.in1d(cortex['tri'][:,0],vertnums))[0]) & set(np.where(np.in1d(cortex['tri'][:,1],vertnums))[0]) & set(np.where(np.in1d(cortex['tri'][:,2],vertnums))[0]))))
        tri_list = cortex['tri'][tri_row_inds,:]
        lookup = {vertnums[i]:i for i in range(len(vertnums))}
       
        tri_list_reindexed = np.copy(tri_list)
        for k, v in lookup.iteritems():
            tri_list_reindexed[tri_list==k] = v

        roi_mesh['tri'] = tri_list_reindexed

        if showfig:
            from .plotting import ctmr_brain_plot as ctmr_brain_plot   
            mesh,mlab = ctmr_brain_plot.ctmr_gauss_plot(roi_mesh['tri'],roi_mesh['vert'])
            mlab.show()

        if save:
            output_mesh = os.path.join(self.mesh_dir,'%s_%s_trivert.mat'%(hem, roi_name))
            print("Saving this mesh to %s"%(output_mesh))
            scipy.io.savemat(output_mesh, roi_mesh)
        
        return roi_mesh
    
    def write_to_obj(self, hem=None, roi_name='pial'):
        '''This function writes the mesh for a given roi to .obj format.
        
        Parameters
        ----------
        hem : str, optional
            The hemisphere of the region of interest (ROI)
        roi_name : str, optional
            The name of the ROI mesh.

        '''
        if hem==None:
            hem = self.hem
        cortex = self.get_surf(roi=roi_name, hem=hem)
        tri, vert = cortex['tri'], cortex['vert']

        f = open(os.path.join(self.mesh_dir,'%s_%s.obj'%(hem, roi_name)),'w+')

        for row in vert:
            f.write('v %f %f %f\n'%(row[0], row[1], row[2]))

        for row in tri:
            f.write('f %d %d %d\n'%(row[0]+1, row[1]+1, row[2]+1))

        f.close()

    def obj_to_mat(self, hem=None, roi_name='pial'):
        '''This function reads in a .obj file and converts it to .mat format
        to be read into img_pipe or matlab.

        Parameters
        ----------
        hem : str, optional
            The hemisphere of the region of interest (ROI)
        roi_name : str, optional
            The name of the ROI mesh.
        '''

        if hem == None:
            hem = self.hem
        vertices=[]
        facelines=[]
        with open(os.path.join(self.mesh_dir, '%s_%s.obj' % (hem, roi_name)), 'r') as f:
            for line in f:
                if line.startswith("v "):
                    varray=line.replace('\n','')
                    varray1=varray.split(' ')
                    vertices.append(varray1[1:4])
                    
                if line.startswith("f "):
                    faceline = line.replace('\n','')
                    faceline1 = faceline.replace('/',' ')
                    farray = faceline1.split(' ')
                    farray = list(filter(None, farray))
                    facelines.append(farray[1:len(farray)])

        facelines = list(filter(None, facelines))

        vert = np.asarray(vertices, dtype=np.float)
        face = np.asarray(facelines, dtype=np.int)
        tri = np.zeros((face.shape[0], 3), dtype=np.int)

        if face.shape[1] == 3:
            tri = face - 1

        elif face.shape[1] == 6:
            tri[:, 0] = face[:, 0] - 1
            tri[:, 1] = face[:, 2] - 1
            tri[:, 2] = face[:, 4] - 1

        elif face.shape[1] == 9:
            tri[:, 0] = face[:, 0] - 1
            tri[:, 1] = face[:, 3] - 1
            tri[:, 2] = face[:, 6] - 1

        out_file_trivert = os.path.join(self.mesh_dir, '%s_%s_trivert.mat'%(hem, roi_name))
        scipy.io.savemat(out_file_trivert, {'tri': tri, 'vert': vert})

        cortex = {'tri': tri+1, 'vert': vert}
        out_file_struct = os.path.join(self.mesh_dir, '%s_%s.mat' % (hem, roi_name))
        scipy.io.savemat(out_file_struct, {'cortex': cortex})

    def convert_bsmesh2mlab(self, mesh_name='pial.cortex'):
        '''This function reads in .dfs surfaces that have been created with brainsuite from the T1.nii volume in the acpc directory.

         mesh_name : str The name of the mesh to convert. string between T1. and .dfs

         pial.cortex is used by default. It contains both hemispheres'''

        #         Header format
        # [000-011]	char headerType[12]; // should be DFS_BE v2.0\0 on big-endian machines, DFS_LEv1.0\0 on little-endian
        # [012-015] int32 hdrsize;			// Size of complete header (i.e., offset of first data element)
        # [016-010] int32 mdoffset;			// Start of metadata.
        # [020-023] int32 pdoffset;			// Start of patient data header.
        # [024-027] int32 nTriangles;		// Number of triangles
        # [028-031] int32 nVertices;		// Number of vertices
        # [032-035] int32 nStrips;			// Number of triangle strips (deprecated)
        # [036-039] int32 stripSize;		// size of strip data  (deprecated)
        # [040-043] int32 normals;			// 4	Int32	<normals>	Start of vertex normal data (0 if not in file)
        # [044-047] int32 uvoffset;			// Start of surface parameterization data (0 if not in file)
        # [048-051] int32 vcoffset;			// vertex color
        # [052-055] int32 labelOffset;	// vertex labels
        # [056-059] int32 vertexAttributes; // vertex attributes (float32 array of length NV)
        # [060-183] uint8 pad2[4 + 15*8]; // formerly 4x4 matrix, affine transformation to world coordinates, now used to add new fields

        #Load the tri and vert data from the Brainsuite dfs file
        with open(os.path.join(self.acpc_dir, 'T1.%s.dfs' % (mesh_name)), 'rb') as f:
            f.seek(12)
            hdrsize = struct.unpack('i', f.read(4))[0]
            f.seek(24)
            nTriangles = struct.unpack('i', f.read(4))[0]
            f.seek(28)
            nVertices = struct.unpack('i', f.read(4))[0]

            f.seek(hdrsize)

            tri = np.zeros((nTriangles, 3), dtype='int32')
            dfsvert = np.zeros((nVertices, 3), dtype='float32')

            for triangle in range(0, nTriangles):
                tri[triangle, 0] = struct.unpack('i', f.read(4))[0]
                tri[triangle, 1] = struct.unpack('i', f.read(4))[0]
                tri[triangle, 2] = struct.unpack('i', f.read(4))[0]

            for vertex in range(0, nVertices):
                dfsvert[vertex, 0] = struct.unpack('f', f.read(4))[0]
                dfsvert[vertex, 1] = struct.unpack('f', f.read(4))[0]
                dfsvert[vertex, 2] = struct.unpack('f', f.read(4))[0]

        #Load in the coordinate and transformation information from the T1 file to apply to the vertices
        T1file=os.path.join(self.acpc_dir, 'T1.nii')
        T1voxdim = os.popen('mri_info %s --res' % T1file).read()
        T1vox2ras = os.popen('mri_info %s --vox2ras' % T1file).read()
        T1cras = os.popen('mri_info %s --cras' % T1file).read()

        #Convert the coordinate system and transformation to usable numpy format
        crassplit = np.array(T1cras.split(), dtype='float')
        vox2raslines = T1vox2ras.splitlines()
        vox2ras = np.empty((4, 4),dtype='float')
        for line in range(0, len(vox2raslines)):
            vox2ras[line, :] = vox2raslines[line].split()

        #Combine origin shift with transformation matrix
        vox2ras[0:3, 3] = vox2ras[0:3, 3] - crassplit[0:3]
        voxdim = np.array(T1voxdim.split(), dtype='float')

        #Divide by voxel size to convert to voxels and apply transformation matrix
        vox = dfsvert / voxdim[0:3]
        pointarray = np.append(vox,np.ones((vox.shape[0],1)),axis=1)
        vert = apply_transform_to_points(pointarray, vox2ras)[:, 0:3]

        #Save surfaces to cortex and triver .mat files for output for matlab and python use
        out_file = os.path.join(self.mesh_dir, 'bs_%s_trivert.mat' % (mesh_name))
        out_file_struct = os.path.join(self.mesh_dir, 'bs_%s_%s.mat' % (self.subj, mesh_name))
        scipy.io.savemat(out_file, {'tri': tri, 'vert': vert})
        cortex = {'tri': tri + 1, 'vert': vert}
        scipy.io.savemat(out_file_struct, {'cortex': cortex})

    def plot_all_surface_rois(self, bgcolor=(0, 0, 0), size=(1200, 900), color_dict=None, screenshot=False, showfig=True,
                              **kwargs):
        """
        Plots all of the surface rois for a given subject. Uses colors from the Freesurfer Color lookup table (LUT)
        by default.
        
        Parameters
        ----------
        bgcolor : tuple
            background color (default is black)
        size : tuple
            figure size
        color_dict: None or dict
            freesurfer roi name -> color (default is freesurfer colormap)
        screenshot : bool
            Whether or not to take a screenshot of the mayavi plot
        showfig : bool
            show figure or not.
        kwargs: goes to ctmr_gauss_plot. e.g. ambient, specular, diffuse, etc.
        """
        
        from mayavi import mlab
        from .plotting.ctmr_brain_plot import ctmr_gauss_plot

        if color_dict is None:
            from .SupplementalFiles import FS_colorLUT
            color_dict = FS_colorLUT.get_lut()

        mlab.figure(fgcolor=(0, 0, 0), bgcolor=bgcolor, size=size)
        rois = self.get_rois()
        for roi in tqdm(rois):
            if 'ctx-' + self.hem + '-' + roi in color_dict:
                mesh = self.make_roi_mesh(roi, [roi], save=False)
                color = np.array(color_dict['ctx-' + self.hem + '-' + roi]) / 255.
                ctmr_gauss_plot(mesh['tri'], mesh['vert'], color=color, new_fig=False, **kwargs)

        if self.hem == 'lh':
            azimuth = 180
        elif self.hem == 'rh':
            azimuth = 0
        else:
            azimuth = 90
        mlab.view(azimuth, elevation=90)

        if screenshot:
            arr = mlab.screenshot(antialiased=True)
            plt.figure(figsize=(20, 10))
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            plt.show()

        if showfig:
            mlab.show()

    def auto_2D_brain(self, hem=None, azimuth=None, elevation=90,
                      elecfile_prefix='TDT_elecs_all', template=None,
                      force=False, brain_file=None, elecs_2D_file=None):
        """Generate 2D screenshot of the brain at a specified azimuth and
        elevation, and return projected 2D coordinates of electrodes at this
        view.

        Some code for projection taken from example_mlab_3D_to_2D
        by S. Chris Colbert <sccolbert@gmail.com>,
        see: http://docs.enthought.com/mayavi/mayavi/auto/example_mlab_3D_to_2D.html
        
        Parameters
        ----------
        hem : {None, 'lh', 'rh', 'both'}
            Hemisphere to show. If None, defaults to self.hem.  
        azimuth : float
            Azimuth for brain plot. Normally azimuth=180 for lh, azimuth=0 for rh
        elevation : float
            Elevation for brain plot
        elecfile_prefix: str
            prefix of the .mat with the electrode coordinates matrix
        template : str or None
            Name of the atlas template (if used instead of the subject's brain)
        force : bool
            Force re-creation of the image and 2D coordinates even if the files
            exist.
        brain_file: (optional) None or str
            Filename used when saving 2D brain image. If None, filename is
            automatically generated based on view orientation.
        elecs_2D_file: (optional) None or str
            Filename used when saving electrode position file. If None,
            filename is automatically generated based on view orientation.
        
        Returnsbr
        -------
        brain_image : array-like
            2D brain image
        elecmatrix_2D : array-like
            elecs x 2 matrix of coordinates in 2D that match brain_image and
            can be plotted using matplotlib pyplot instead of mayavi.
        
        """
        from PIL import Image

        if hem is None:
            roi_name = self.hem + '_pial'
        elif hem == 'lh' or hem == 'rh':
            roi_name = hem + '_pial'
        else:
            roi_name = 'pial'

        if azimuth is None:
            if self.hem == 'lh':
                azimuth = 180
            elif self.hem == 'rh':
                azimuth = 0
            else:
                azimuth = 90

        # Add the template to the file name if it is specified, also
        # don't save the same template view over and over if it isn't
        # necessary
        if template is not None:
            template_nm = template
            mesh_dir = os.path.join(self.subj_dir, template, 'Meshes')
        else:
            template_nm = ''
            mesh_dir = self.mesh_dir

        # Path to each 2D file (a screenshot of the brain at a given angle,
        # as well as the 2D projected electrode coordinates for that view).
        if brain_file is None:
            brain_file = os.path.join(mesh_dir, 'brain2D_az%d_el%d%s.png' %
                                      (azimuth, elevation, template_nm))
        if elecs_2D_file is None:
            elecs_2D_file = os.path.join(self.elecs_dir,
                                         '%s_2D_az%d_el%d%s.mat' %
                                         (elecfile_prefix, azimuth, elevation,
                                          template_nm))

        # Test whether we already made the brain file
        if os.path.isfile(brain_file) and force is False:
            if not os.path.isfile(elecs_2D_file):
                raise ValueError(
                    'If brain_file is an existing file and force is False, '
                    'elecs_2D_file must also be an existing file.')
            # Get the file
            # print("Loading previously saved file %s"%(brain_file))
            im = Image.open(brain_file)
            brain_image = np.asarray(im)

        else:
            # Get pial surface and plot it at specified azimuth and elevation
            pial = self.roi(name=roi_name)
            mesh, points, mlab, brain_image, f = \
                self.plot_brain(rois=[pial], screenshot=True, showfig=False,
                                helper_call=True, azimuth=azimuth,
                                elevation=elevation, template=template)

            # Clip out the white space (there may be a better way to do this..)
            brain_image, x_offset, y_offset = remove_whitespace(brain_image)

            # Save as a png
            im = Image.fromarray(brain_image)
            im.save(brain_file)

        # Test whether we already have the electrodes file
        if os.path.isfile(elecs_2D_file) and force is False:
            #print("Loading previously saved file %s"%(elecs_2D_file))
            elecmatrix_2D = scipy.io.loadmat(elecs_2D_file)['elecmatrix']

        else:
            # Get the 3D electrode matrix
            e = self.get_elecs(elecfile_prefix=elecfile_prefix)
            elecmatrix = e['elecmatrix']
            
            W = np.ones(elecmatrix.shape[0])
            hmgns_world_coords = np.column_stack((elecmatrix, W))

            # Get unnormalized view coordinates
            combined_transform_mat = get_world_to_view_matrix(f.scene)
            view_coords = \
                apply_transform_to_points(hmgns_world_coords,
                                          combined_transform_mat)

            # Get normalized view coordinates
            norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

            # Transform from normalized coordinates to display coordinates (2D)
            view_to_disp_mat = get_view_to_display_matrix(f.scene)
            disp_coords = apply_transform_to_points(norm_view_coords,
                                                    view_to_disp_mat)
            elecmatrix_2D = np.zeros((elecmatrix.shape[0], 2))
            for i in np.arange(elecmatrix.shape[0]):
                elecmatrix_2D[i, :] = disp_coords[:, :2][i]

            elecmatrix_2D[:, 0] = elecmatrix_2D[:, 0] - x_offset
            elecmatrix_2D[:, 1] = elecmatrix_2D[:, 1] - y_offset

            print(elecmatrix_2D)

            scipy.io.savemat(elecs_2D_file, {'elecmatrix': elecmatrix_2D})
            mlab.close()

        return brain_image, elecmatrix_2D

    def animate_scene(self, mlab, movie_name='movie', mix_type='smootherstep', 
                      start_stop_azimuth=(0, 360), start_stop_elevation=(90, 90), 
                      nframes=200, frame_rate=25, ffmpeg = '/Applications/ffmpeg',
                      close_fig=True, keep_frames=False, show_title=False):
        ''' Create animation of rotating brain 

        Parameters
        ----------
        mlab : mayavi mlab instance
        movie_name : str
            Name of the movie frame e.g. 'TIMIT_activity'
        mix_type : {'smootherstep', 'smoothstep', 'linear'}
            Easing for the movie.  If 'smootherstep' or 'smoothstep', the animation
            appears to go slowly at the beginning and end, and fast in the middle.
            If 'linear', it evenly rotates with no easing.
        start_stop_azimuth : tuple
            Start and stop azimuth for the plot. Usually you will want this to be
            (0, 360) for 'rh' subjects and (180, 180+360) for 'lh' subjects
        start_stop_elevation : tuple
            Start and stop azimuth for the plot
        nframes : int
            Number of frames in the animation.  More frames = smoother animation
        frame_rate : int
            The frame rate for the movie in Hz
        ffmpeg : str
            Location of your ffmpeg binary
        close_fig : bool
            Whether to close the mlab figure after the movie is done rendering
        keep_frames : bool
            Whether to keep the pngs for each frame of the movie or delete them
            upon mp4 creation. (default: False)
        show_title : bool
            Whether to keep the current mlab figure title on the screen or suppress it
            (Default: False)

        Returns
        -------
        A movie file in $SUBJECTS_DIR/[self.subj]/movies/[movie_name].mp4

        '''
        movie_dir = os.path.join(self.patient_dir, 'movies')
        mix = mixes[mix_type]

        if not os.path.isdir(movie_dir):
            os.mkdir(movie_dir)
        if not os.path.isdir(os.path.join(movie_dir, 'frames')):
            os.mkdir(os.path.join(movie_dir, 'frames'))

        if show_title==False:
            mlab.title('')

        plt.figure(figsize=(20,10))
            
        for mi, m in enumerate(np.linspace(0, 1, nframes)):
            new_az = mix(start_stop_azimuth[0], start_stop_azimuth[1], m)
            new_el = mix(start_stop_elevation[0], start_stop_elevation[1], m)
            print(new_az, new_el)
            mlab.view(new_az, new_el)
            arr = mlab.screenshot(antialiased=True)
            plt.cla()
            plt.imshow(arr, aspect='equal')
            plt.axis('off')
            frame_png = os.path.join(movie_dir, 'frames', '%s_%03d.png'%(movie_name, mi))
            plt.savefig(frame_png)

        png_files = os.path.join(movie_dir, 'frames', movie_name+'_%03d.png')
        movie_file = os.path.join(movie_dir, movie_name+'.mp4')
        os.system('%s -r %d -start_number 0 -i %s \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 \
            -pix_fmt yuv420p %s'%(ffmpeg, frame_rate, png_files, movie_file))

        if not keep_frames:
            png_files = os.path.join(movie_dir, 'frames', movie_name+'_*.png')
            [os.remove(r) for r in glob.glob(png_files)]

        if close_fig:
            mlab.close()


# Functions #

def remove_whitespace(brain_image):
    '''
    Remove white space from an image

    Parameters
    ----------
    brain_image : array-like
        X x Y x 3 RGB image

    Returns
    -------
    brain_image : array-like
        X x Y x 3 RGB image, but trimmed
    x_offset : int
        Number of pixels to offset
    y_offset : int
        Number of pixels to offset
    '''
    rgb_sum = brain_image.sum(2)
    white_space1 = 1-np.all(rgb_sum==255*3, axis=0)
    x_offset = np.where(white_space1==1)[0][0]
    brain_image = brain_image[:,white_space1.astype(bool),:]
    white_space2 = 1-np.all(rgb_sum==255*3, axis=1)
    y_offset = np.where(white_space2==1)[0][0]
    brain_image = brain_image[white_space2.astype(bool),:,:]

    return brain_image, x_offset, y_offset

# Private Functions
def _str2bool(v):
    ''' Changes a string to a boolean.'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _method_args_to_parser(method, parser):
    '''Adds argument information from method to a parser'''
    method_argnames = inspect.getargspec(method)[0]
    method_argnames.pop(0)
    defaults = inspect.getargspec(method)[3]
    if defaults:
        required_args = len(method_argnames) - len(defaults)
        args_values = [None] * required_args + list(defaults)
    else:
        required_args = len(method_argnames)
    for arg_number in range(len(method_argnames)):
        if arg_number < required_args:
            parser.add_argument(method_argnames[arg_number])
        elif args_values[arg_number] is None:
            parser.add_argument('--' + method_argnames[arg_number], default=args_values[arg_number])
        elif type(args_values[arg_number]) == bool:
            parser.add_argument('--' + method_argnames[arg_number], type=_str2bool, default=args_values[arg_number])
        else:
            parser.add_argument('--' + method_argnames[arg_number], type=type(args_values[arg_number]), default=args_values[arg_number])


########## Main ##########

if __name__ == "__main__":
    # Get the argument information of the freeCoG methods
    free_methods = inspect.getmembers(freeCoG, predicate=inspect.ismethod)
    init_method = free_methods.pop(0)

    # Build parser
    parser = argparse.ArgumentParser(description=('img_pipe '))
    _method_args_to_parser(init_method[1],parser)
    subparsers = parser.add_subparsers(dest='method',metavar='method')
    # Create subparser for each freeCoG method
    for free_method in free_methods:
        method_parser = subparsers.add_parser(free_method[0], help=free_method[1].__doc__)
        # Get arguments and defaults for the freeCoG method
        _method_args_to_parser(free_method[1],method_parser)


    # Parse Commandline arguments
    args = parser.parse_args(sys.argv[1:])
    freeCoG_args = {}
    method_args = vars(args)
    # Remove extra spaces from arguments and set None types
    for key in method_args:
        if type(method_args[key]) == str:
            method_args[key] = method_args[key].lstrip()
        if method_args[key] == 'None':
            method_args[key] = None
    # Separate out initialization arguments and instantiate freeGoG
    for key in inspect.getargspec(init_method[1])[0][1:]:
        freeCoG_args[key] = method_args[key]
        del method_args[key]
    print('Instantiating freeCoG with:')
    print(freeCoG_args)
    patient = freeCoG(**freeCoG_args)
    # Get the method, its arguments, and run it
    freeCoG_method = getattr(patient, args.method)
    print('Running %s with arguments:' % (method_args['method']))
    del method_args['method']
    print(method_args)
    freeCoG_method(**method_args)
