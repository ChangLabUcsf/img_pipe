from setuptools import setup, find_packages
from setuptools.command.install import install
import os, subprocess

class MyInstall(install):

    def run(self):
        install.run(self)
        os.system("./img_pipe/dependencies.sh")

setup(name = "img_pipe",
	  description = "Image processing pipeline for localization and identification of electrodes for electrocorticography",
	  version = "2017.6.16.1",
	  url = "https://github.com/ChangLabUcsf/img_pipe",
	  author = "Liberty Hamilton",
	  author_email = "libertyhamilton@gmail.com",
	  packages = find_packages(),
	  include_package_data = True,
	  setup_requires=['cython','numpy','scipy'],
	  install_requires=['numpy','scipy','pyvtk','mayavi','pymcubes','mne','nibabel','nipy','matplotlib<2.0.0','configparser','tqdm'],
	  dependency_links=['http://www.vtk.org/files/release/6.3/VTK-6.3.0.tar.gz','https://downloads.sourceforge.net/project/pyqt/PyQt4/PyQt-4.12/PyQt4_gpl_mac-4.12.tar.gz?r=&ts=1487961590&use_mirror=superb-sea2'],
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
