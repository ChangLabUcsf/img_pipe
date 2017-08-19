#!/usr/bin/env python
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Just test whether the import works
import img_pipe
#from img_pipe import freeCoG
patient = img_pipe.freeCoG(subj = 'S1', hem = 'lh', fs_dir='', subj_dir='.')

# Test whether mayavi works
import mayavi
from mayavi import mlab

mlab.test_mesh()

# Test import of electrode picker
from .SupplementalScripts import electrode_picker

# Test import of plotting
from .plotting import ctmr_brain_plot

