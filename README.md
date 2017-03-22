## ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/leftbrain_blackbg.png "img_pipe") img_pipe: Image processing pipeline for ECoG data ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/rightbrain_blackbg.png "img_pipe") ##


Developed by Liberty Hamilton, David Chang, Morgan Lee at the Laboratory of Dr. Edward Chang, UC San Francisco
http://changlab.ucsf.edu

This contains the imaging pipeline as one importable python class for running a patient's
brain surface reconstruction and electrode localization/labeling.

## About ##
Show end results and capabilities, workflow diagram.

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

You should now be able to import img_pipe from python. 
```python
>>> import img_pipe
>>> patient = img_pipe.freeCoG(subj='subject_name', hem='lh')
>>> patient.prep_recon()
>>> patient.plot_recon_anatomy()
```

## ChangLab Specific: ##
Download OSXFUse
add to ~/.bash_profile 

```
alias duramount='sshfs -p 7777 -o defer_permissions dchang@dura.cin.ucsf.edu:/ /Users/dlchang/dura'

alias duraunmount='sudo umount /Users/dlchang/dura/'
```




