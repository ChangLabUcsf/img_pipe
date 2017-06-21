## ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/leftbrain_blackbg.png "img_pipe") img_pipe: Image processing pipeline for ECoG data ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/rightbrain_blackbg.png "img_pipe") ##


Developed by Liberty Hamilton, David Chang, Morgan Lee at the Laboratory of Dr. Edward Chang, UC San Francisco
http://changlab.ucsf.edu

This contains the imaging pipeline as one importable python class for running a patient's
brain surface reconstruction and electrode localization/labeling.

The full capabilities of the pipeline are described in the paper: 
Hamilton LS, Chang DL, Lee MB, Chang EF. Semi-automated anatomical labeling and inter-subject warping of high-density intracranial recording electrodes in electrocorticography.

## About ##
`img_pipe` is an open source python package for preprocessing of imaging data for use in intracranial electrocorticography (ECoG) and intracranial stereo-EEG analyses. This python package aims to provide a standardized interface for electrode localization, labeling, and warping to an atlas, as well as code to plot and display results on 3D cortical surface meshes. It gives the user an easy interface to create anatomically labeled electrodes that can also be warped to an atlas brain, starting with only a preoperative T1 MRI scan and a postoperative CT scan. 

Example results are shown below in the native subject space (left) and in the cvs_avg35_inMNI152 atlas space (right):

![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalFiles/img_pipe_results.png "img_pipe")

## Setup and Installation ##

To download this package, you will need:
* a MacOS or Linux machine (if you are using Windows, download a Linux Virtual Machine to use this package)
* __anaconda__ for Python version 2.7, not 3 (https://www.continuum.io/downloads)<br>
* __Freesurfer__ (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) version 5.3.0 or higher

After you download and install those dependencies, run the following commands in your terminal:

``` 
$ conda install vtk
$ conda install pyqt==4.11.4 
$ pip install img_pipe
 ```

After that, edit your ~/.bash_profile or ~/.bashrc and set the following environment variables with these lines:

```
export SUBJECTS_DIR=/path/to/freesurfer/subjects
export FREESURFER_HOME=/path/to/freesurfer/
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
Note that you can set `SUBJECTS_DIR` to wherever you want to place your subjects' imaging data - for example, `/Applications/freesurfer/subjects`.

Then in terminal, run `source ~/.bash_profile` or `source ~/.bashrc` to activate these environment variables.

To run `img_pipe`, you will need a high quality non-contrast T1 scan and a non-contrast CT scan. The T1 scan should ideally be 
AC-PC aligned before you start. Name the T1 scan T1.nii and place in `$SUBJECTS_DIR/your_subj/acpc`.  Name the CT scan CT.nii 
and place in `$SUBJECTS_DIR/your_subj/CT`.


You should now be able to import img_pipe from python. 
```python
>>> import img_pipe
>>> patient = img_pipe.freeCoG(subj='subject_name', hem='lh')
>>> patient.prep_recon()
>>> patient.get_recon()
```

If you have completed all of the steps, you can plot the brain with anatomically-labeled electrodes as follows:
```python
>>> import img_pipe
>>> patient = img_pipe.freeCoG(subj='subject_name', hem='lh')
>>> patient.plot_recon_anatomy()
```

Or just the brain with
```python
>>> patient.plot_brain()
```

The full workflow is shown as a flowchart below:

![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalFiles/workflow.png "img_pipe")

If you find any bugs, please post in the Issues tab. 
