#plot_recon_anatomy.py
#
# Liberty Hamilton 2016
#

import mayavi
import scipy.io
import ctmr_brain_plot
from ../SupplementalFiles import FS_colorLUT
import numpy as np
import os

fs_dir = os.environ['SUBJECTS_DIR']

def plot_recon_anatomy(patient):
	subj = patient.subj
	hem = patient.hem
	a = scipy.io.loadmat('%s/%s/Meshes/%s_pial_trivert.mat'%(fs_dir, subj, hem))
	e = scipy.io.loadmat('%s/%s/elecs/TDT_elecs_all.mat'%(fs_dir, subj))

	# Plot the pial surface
	mesh, mlab = ctmr_brain_plot.ctmr_gauss_plot(a['tri'], a['vert'], color=(0.8, 0.8, 0.8))
	
	# Add the electrodes, colored by anatomical region
	elec_colors = np.zeros((e['elecmatrix'].shape[0], e['elecmatrix'].shape[1]))

	# Import freesurfer color lookup table as a dictionary
	cmap = FS_colorLUT.get_lut()

	# Make a list of electrode numbers
	elec_numbers = np.arange(e['elecmatrix'].shape[0])+1

	# Find all the unique brain areas in this subject
	brain_areas = np.unique(e['anatomy'][:,3])

	# Loop through unique brain areas and plot the electrodes in each brain area
	for b in brain_areas:
	    # Add relevant extra information to the label if needed for the color LUT
	    this_label = b[0]
	    if b[0][0:3]!='ctx' and b[0][0:4] != 'Left' and b[0][0:5] != 'Right' and b[0][0:5] != 'Brain' and b[0] != 'Unknown':
	        this_label = 'ctx-%s-%s'%(hem, b[0])
	        print(this_label)
	   	
	    if this_label != '':
	        el_color = np.array(cmap[this_label])/255.
	        ctmr_brain_plot.el_add(np.atleast_2d(e['elecmatrix'][e['anatomy'][:,3]==b,:]), 
	        					   color=tuple(el_color), numbers=elec_numbers[e['anatomy'][:,3]==b])
	mlab.show()        
	return mesh, mlab
