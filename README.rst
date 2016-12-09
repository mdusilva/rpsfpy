rpsfpy: GLAO PSF reconstruction in Python

This is a set of tools to reconstruct PSFs of GLAO systems, taking 
into account the anisoplanetism effect.

Contributors: M. Silva, T. Fusco, R. Villecroze, B. Bneichel, R Bacon,
P-Y Madec, M. Lelouran, A. Guesalaga

-------------------------
Usage example:

>>> import numpy as np
>>> import rpsfpy

#Sctructure class instance

>>> structure = rpsfpy.Structure() # can also give file as input:
>>> structure = rpsfpy.Structure('foobar.cfg')

#Object position in the field in arcsec

>>> objpos = np.array(([[0.,0.]]))

#Lambda in nm

>>> lambdaim = 600

#compute PSF

>>> psf = structure.psf(objpos, lambdaim, out='psf.fits')

#By default the program will run in parallel using all available cores.
#To specify the number of cores do:

>>> psf = structure.psf(objpos, lambdaim, out='psf.fits', parallel=2) #2 cores


The parameters included in the configuration file are the atmospheric parameters, 
AO system parameters and image parameters.

The contents of the default configuration file ("default.cfg") are:

[atmosphere]
r0 = 41.475
cn2 profile = 0.7,0.3
h profile = 0.0,10000.0
l0 = 25

[AO system]
ngs = 0., 0.
lgs = 32.0,32.0,32.0,-32.0,-32.0,32.0,-32.0,-32.0
lgs altitude = 90000.
zernike number = 60
telescope diameter = 8

[image]
pixel diameter = 128

In order to use a different configuration file, copy these contents to a new file
and edit as desired.
