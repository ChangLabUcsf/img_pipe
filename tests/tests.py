#!/usr/bin/python

import img_pipe
import os
os.environ['FREESURFER_HOME']=''
os.environ['SUBJECTS_DIR']=''

patient = img_pipe.freeCoG(subj = 'test', hem = 'lh')
