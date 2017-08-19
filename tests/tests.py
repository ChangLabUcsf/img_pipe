#!/usr/bin/env python

# Just test whether the import works
import img_pipe
from img_pipe import freeCoG
patient = freeCoG(subj = 'S1', hem = 'lh', fs_dir='', subj_dir='tests')

# Test import of electrode picker
from . import SupplementalScripts, plotting
from SupplementalScripts import electrode_picker

# Test import of plotting
from plotting import ctmr_brain_plot

# Test whether mayavi works
import mayavi
from mayavi import mlab

mlab.test_mesh()
