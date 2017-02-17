from setuptools import setup, find_packages

setup(name = "img_pipe",
	  description = "Image processing pipeline for localization and identification of electrodes for electrocorticography",
	  version = "2017.2.16",
	  url = "https://github.com/ChangLabUcsf/img_pipe",
	  author = "Liberty Hamilton",
	  author_email = "libertyhamilton@gmail.com",
	  packages = find_packages(),
	  include_package_data = True,
	  classifiers = [
	  	"Intended Audience :: Science/Research",
	  	"Intended Audience :: Education",
	  	"Programming Language :: Python :: 2.7",
	  	"Topic :: Scientific/Engineering",
	  	"Topic :: Scientific/Engineering :: Visualization",
	  	"Topic :: Scientific/Engineering :: Medical Science Apps."
	  	]
)