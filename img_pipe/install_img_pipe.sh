#!/usr/bin/env bash

#get homebrew to download dcmtk and dcm2niix
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

echo -ne $password | brew install dcm2niix

#check if freesurfer exists and build it if not
export FREESURFER_HOME="${FREESURFER_HOME:-/Applications/freesurfer/}"
if [ ! -f $FREESURFER_HOME/build-stamp.txt ]; then
    if [ ! -f ~/Downloads/freesurfer-Darwin-OSX-stable-pub-v6.0.0.dmg ]; then
        curl -o ~/Downloads/freesurfer-Darwin-OSX-stable-pub-v6.0.0.dmg ftp://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Darwin-OSX-stable-pub-v6.0.0.dmg
    fi

    echo -ne $password | sudo hdiutil attach ~/Downloads/freesurfer-Darwin-OSX-stable-pub-v6.0.0.dmg
    echo -ne "$password $password" | sudo installer -pkg /Volumes/freesurfer-Darwin-full/freesurfer-Darwin-full.pkg -target /
    echo -ne $password | sudo hdiutil detach /Volumes/freesurfer-Darwin-full
    if [ -f $FREESURFER_HOME/build-stamp.txt ]; then
        rm ~/Downloads/freesurfer-Darwin-OSX-stable-pub-v6.0.0.dmg
    fi
fi

#install miniconda. assume that no anaconda python
curl 'https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh' > miniconda3.sh
bash miniconda3.sh
echo -ne "yes" | /Users/${USER}/miniconda3/bin/conda update conda

#create conda environment for img_pipe and activate environment
/Users/${USER}/miniconda3/bin/conda create -n img_pipe python=2.7
/Users/${USER}/miniconda3/bin/conda activate img_pipe

#install prerequisite modules in img_pipe environment
/Users/${USER}/miniconda3/envs/img_pipe/bin/pip install numpy
/Users/${USER}/miniconda3/envs/img_pipe/bin/pip install vtk
/Users/${USER}/miniconda3/condabin/conda install pyqt==4.11.4 --name img_pipe

#install img_pipe using pip install
/Users/${USER}/miniconda3/envs/img_pipe/bin/pip install img_pipe