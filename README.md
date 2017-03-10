# img_pipe
![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/icons/leftbrain_blackbg.png "img_pipe") Image processing pipeline for ECoG data

Developed by Liberty Hamilton, David Chang, Morgan Lee

This contains the imaging pipeline as one importable python class for running a patient's
brain surface reconstruction and electrode localization/labeling.

To download this package, you will need:

<b> anaconda </b> (https://www.continuum.io/downloads)<br>
<b> pip </b> (you can get this by running <i>sudo easy_install pip</i> or <i>sudo apt-get pip</i> or download from https://pip.pypa.io/en/stable/installing/<br>
<b> MATLAB </b>

Afterwards, run the following two commands:<br>
<i>conda install vtk</i><br>
<i>conda install pyqt==4.11.4</i><br>

Then, run <b>pip install img_pipe</b> from the terminal. 

You can then download spm (http://www.fil.ion.ucl.ac.uk/spm/software/download/), and Freesurfer (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). 

After that, edit your ~/.bash_profile or ~/.bashrc and set the following environment variables with these lines:

export FREESURFER_HOME=/path/to/freesurfer/<br>
export SUBJECTS_DIR=/path/to/subjects/directory/<br>
export SPM_PATH=/path/to/spm/<br>

(matlab environment variable?)

You should now be able to import img_pipe from python. 

\>>> import img_pipe<br>
\>>> patient=img_pipe.freeCoG(subj='subject_name',hem='lh')<br>
\>>> patient.prep_recon()<br>



