from setuptools import setup, find_packages
from setuptools.command.install import install
import os, subprocess

class MyInstall(install):

    def run(self):
        install.run(self)
        os.system("./img_pipe/dependencies.sh")

setup(name = "img_pipe",
	  description = "Image processing pipeline for localization and identification of electrodes for electrocorticography",
	  version = "2017.2.24.4",
	  url = "https://github.com/ChangLabUcsf/img_pipe",
	  author = "Liberty Hamilton",
	  author_email = "libertyhamilton@gmail.com",
	  packages = find_packages(),
	  include_package_data = True,
	  #setup_requires=['numpy','scipy'],
	  install_requires=[],#['numpy','scipy','mayavi','pymcubes','pysurfer','mne','nibabel','nipy','nipype'],
	  dependency_links=['https://downloads.sourceforge.net/project/pyqt/PyQt4/PyQt-4.12/PyQt4_gpl_mac-4.12.tar.gz?r=&ts=1487961590&use_mirror=superb-sea2'],
	  cmdclass={'install':MyInstall},
	  classifiers = [
	  	"Intended Audience :: Science/Research",
	  	"Intended Audience :: Education",
	  	"Programming Language :: Python :: 2.7",
	  	"Topic :: Scientific/Engineering",
	  	"Topic :: Scientific/Engineering :: Visualization",
	  	"Topic :: Scientific/Engineering :: Medical Science Apps."
	  	]
)