"""
RPSFpy - PSF reconstruction for GLAO

This is a set of tools to reconstruct PSFs of GLAO systems, taking
into account the anisoplanetism effect.

Method
------
The PSF estimation method is the one described in [1]. The basic idea is to approximate the 
residual phase in WFM considering the most dominant source of variance, the effect of anisoplanatism.
The PSF is estimated via the OTF computed from the structure function. The structure function is approximated 
by the sum of three components, where only the effect of anisoplanetism and the fitting error are considered. 
These effects are related to the AO correction by LGSs and one NGS.


Features
--------
1. Compute PSFs (or just the OTFs) in arbitrary positions in the Field of View
2. Computation of the LGS structure function is done in parallel
3. Several terms of the LGS structure function and the Fitting error structure function are stored in a library 
to speed up posterior computations

Dependencies
------------
This software only runs on Linux systems.
It has the following dependencies:

1. a Python 2.7 distribution;
2. Numpy package;
3. Scipy package;
4. pyfits OR astropy package.

We suggest the use of the Anaconda Python platform as it includes all the necessary dependencies. Download at <https://www.continuum.io/downloads>

Installation
------------
The software is distributed in the form of a Python package which can be downloaded from its public repository at: 
 
<https://github.com/mdusilva/rpsfpy>

We suggest that it should be installed in its own virtual environment. To create and use a virtual environment in the Anaconda distribution 
follow the instructions in <https://conda.io/docs/using/envs.html>
 
To install the package (instructions for Linux):
 
create a new temporary directory in your home directory:

    
    #!bash
    mkdir rpsf


download the contents of the repository to the new directory. The directory should now contain a file named **setup.py** and a folder **rpsfpy**.
in the terminal, execute the command:


    #!bash
    python setup.py install


you may now remove the temporary directory


    #!bash
    cd ..
    rm -r rpsf


Input
-----
The following inputs are needed to compute one or more PSFs:

1. atmospheric parameters: r0 (at 500 nm), cn2 profile (normalized) and respective height profile (in m) and 
outer_scale (in m). These parameters are passed as an `rpsfpy.rpsf.Atmosphere` object.
2. parameters related to the AO system: LGSs coordinates, LGSs height, number of corrected Zernike modes and
the telescope diameter. These parameters are stored together in the Python dictionary `rpsfpy.rpsf.ao_systems` 
labeled by the AO system name which is created and managed by `rpsfpy.rpsf.AOsystem`.
Note that the PSF recontruction method was created with the specific case of MUSE-GALACSI in mind and may not work 
properly with other AO systems.
3. The position of the NGS(s) in the FoV (in arcsec) and the image sampling (diameter of the telescope pupil 
in pixels), passed to a `rpsfpy.rpsf.Reconstruct` object. Finally, the coordinates in the FoV (in arcsec) where 
to compute the PSFs and wavelength (in nm), passed to `rpsfpy.rpsf.Reconstruct.psf` method. 


Computing PSFs
--------------
To compute one or more PSFs do the following:

Import this package and Numpy


    #!python
    >>> import rpsfpy
    >>> import numpy as np

Define the Cn2 profile and height of the respective atmospheric layers (in this example we have two layers)


    #!python
    >>> cn2 = [0.7,0.3]
    >>> h = [0.,10000.]


Create an instance of an `rpsfpy.rpsf.Atmosphere` object for a seeing of 0.8" at 500 nm, with 
an outer scale of 25 m:


    #!python
    >>> a = rpsfpy.Atmosphere(rzero=0.12634, cn2_profile=cn2, h_profile=h,outer_scale=25.)


Create a `rpsfpy.rpsf.Reconstruct` instance considering a GALACSI like `rpsfpy.rpsf.AOsystem`, the atmosphere defined 
by the previous `rpsfpy.rpsf.Atmosphere` instance, images with a size of 256 pixels (the image size is twice the diameter of the pupil) 
and one NGS in the (50,0) direction, in arcsec, for tip-tilt correction. Note that the sampling is always done (by default) at the Nyquist rate.


    #!python
    >>> r = rpsfpy.Reconstruct(pixdiam=128, ao_system="GALACSI", ngspos=[[50.,0.]], atmosphere=a)


Create list of positions (in arcsec) where to compute the PSF:


    #!python
    >>> objects = [[0.,0.], [0., 32.]]


Finally, compute the PSF in those postions using the `rpsfpy.rpsf.Reconstruct.psf` method in the `rpsfpy.rpsf.Reconstruct` instance: 


    #!python
    >>> r.psf(objects, 640., out="psf.fits")


This will compute the PSF for the positions contained in the objects list, for a wavelength of 640 nm and output the result 
to a **psf.fits** FITS file.

Computing PSFs with higher sampling rate
----------------------------------------
It is possible to compute PSFs with a sampling rate higher than the default Nyquist limit. To do it use the 
**scale** keyword argument when calling the `rpsfpy.rpsf.Reconstruct.psf` method:


    #!python
    >>> r.psf(objects, 640., scale=1.5, out="psf.fits")


This example produces a sampling rate 1.5 times the Nyquist rate.


Adding new AO systems
---------------------
AO systems with different parameters can be added to the database (a Python dictionary). For example to add a new system composed of 
three LGSs in a triangular configuration correcting 1000 zernike modes do:

First create instance of the `rpsfpy.rpsf.AOsystem` object


    #!python
    >>> aosystems = rpsf.AOsystem()


then use the `rpsfpy.rpsf.AOsystem.add` method


    #!python
    >>> aosystems.add(name="new_AO", lgspos=[[0.,32.], [-32.,-32.], [32.,-32.]], lgsheight=90000., zmodes=1000, diameter=8.)


This will create a new entry in the dictionary with the key 'new_AO' (the name we are giving the AO system). 
Note that we must pass the parameters lgspos, a list of the directions of the LGSs of the AO system (in arcsec) 
lgsheight, the height of the LGSs (in m); zmodes, the number of zernike modes corrected by the AO system; 
and diameter, the diameter of the telescope (in m).

To view the AO systems that currently available for PSF computations do:


    #!python
    >>> aosystems.view()


which will give the output:

    #!python
    new_AO
    ------------
    LGSs coordinates =  [[0.0, 32.0], [-32.0, -32.0], [32.0, -32.0]]
    LGSs height =  90000.0
    Corrected Zernike modes =  1000
    Telescope diameter =  8.0
    ------------
    GALACSI
    ------------
    LGSs coordinates =  [[32.0, 32.0], [32.0, -32.0], [-32.0, 32.0], [-32.0, -32.0]]
    LGSs height =  90000.0
    Corrected Zernike modes =  980
    Telescope diamete



--------------
[1] Villecroze, R., "Modelisation d'un systeme d'optique adaptative a
grand champ pour la reconstruction de la reponse
impulsionnelle multi-spectrale des futurs
spectro-imageurs 3D du VLT et de l'ELT",  Lyon: Universite Claude Bernard Lyon 1, 2014.
"""
from rpsf import polar2, Atmosphere, Reconstruct, StructureNGS, StructureLGS, AOsystem, StructureFit
import zernike
import copy_reg
from types import MethodType

__version__ = '0.3.2'

def _pickle_method(method):
      func_name = method.im_func.__name__
      obj = method.im_self
      cls = method.im_class
      if func_name.startswith('__') and not func_name.endswith('__'):
	cls_name = cls.__name__.lstrip('_')
	if cls_name: func_name = '_' + cls_name + func_name
      return _unpickle_method, (func_name, obj, cls)
    
def _unpickle_method(func_name, obj, cls):
      for cls in cls.mro():
	try:
	  func = cls.__dict__[func_name]
	except KeyError:
	  pass
	else:
	  break
      return func.__get__(obj, cls)
copy_reg.pickle(MethodType,_pickle_method, _unpickle_method)
