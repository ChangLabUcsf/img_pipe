## ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/leftbrain_blackbg.png "img_pipe") img_pipe: Image processing pipeline for ECoG data ![alt text](https://github.com/ChangLabUcsf/img_pipe/raw/master/img_pipe/SupplementalScripts/icons/rightbrain_blackbg.png "img_pipe") ##


Developed by Liberty Hamilton, David Chang, Morgan Lee

This contains the imaging pipeline as one importable python class for running a patient's
brain surface reconstruction and electrode localization/labeling.

To download this package, you will need:
* a MacOS or Linux machine (if you are using Windows, download a Linux Virtual Machine to use this package)
* __anaconda__ (https://www.continuum.io/downloads)<br>
* __MATLAB__
* __Freesurfer__ (https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) version 5.3.0 or higher

After you download and install those dependencies, run the following commands in your terminal:<br>
``` 
$ conda install vtk
$ conda install pyqt==4.11.4 
$ pip install img_pipe
 ```

After that, edit your ~/.bash_profile or ~/.bashrc and set the following environment variables with these lines:

```
export FREESURFER_HOME=/path/to/freesurfer/
export SUBJECTS_DIR=/path/to/subjects/directory/
export SPM_PATH=/path/to/spm/
matlab environ variable?
```
Then in terminal, run `source ~/.bash_profile` or `source ~/.bashrc` to activate these environment variables.

You should now be able to import img_pipe from python. 
```python
>>> import img_pipe
>>> patient = img_pipe.freeCoG(subj='subject_name', hem='lh')
>>> patient.prep_recon()
```



