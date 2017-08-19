#!/usr/bin/python

# Just test whether the import works
import img_pipe
patient = img_pipe.freeCoG(subj = 'S1', hem = 'lh', fs_dir='', subj_dir='tests')

# Test import of electrode picker
from ..SupplementalScripts import electrode_picker

# Test import of plotting
from ..plotting import ctmr_brain_plot
