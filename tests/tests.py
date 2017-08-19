#!/usr/bin/env python
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
print(sys.path)

# Just test whether the import works
import img_pipe
#from img_pipe import freeCoG
patient = img_pipe.freeCoG(subj = 'S1', hem = 'lh', fs_dir='', subj_dir='.')

# Test import of electrode picker
from SupplementalScripts import electrode_picker

# Test import of plotting
from plotting import ctmr_brain_plot

# Test whether mayavi works
import mayavi
from mayavi import mlab

mlab.test_mesh()
