#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

# Just test whether the import works
import img_pipe
#from img_pipe import freeCoG
patient = img_pipe.freeCoG(subj = 'S1', hem = 'lh', fs_dir='', subj_dir='tests')

# Test import of electrode picker
from img_pipe.SupplementalScripts import electrode_picker

# Test import of plotting
from img_pipe.plotting import ctmr_brain_plot

# Test whether mayavi works
import mayavi
from mayavi import mlab

mlab.test_mesh()
