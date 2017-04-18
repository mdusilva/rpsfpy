rpsfpy: GLAO PSF reconstruction in Python

This is a set of tools to reconstruct PSFs of GLAO systems, taking 
into account the anisoplanetism effect.

Contributors: M. Silva, T. Fusco, R. Villecroze, B. Bneichel, R Bacon,
P-Y Madec, M. Lelouran, A. Guesalaga

-------------------------
Usage example:

>>> import numpy as np
>>> import rpsfpy

#height of atmospheric layers

>>> h = [0.,10000.]

#Cn2 values (relative)

>>> cn2 = [0.7,0.3]

#Instantiate atmosphere object for seeing = 0.8''

>>> a = rpsfpy.Atmosphere(rzero=0.12634, cn2_profile=cn2, h_profile=h,outer_scale=25.)

#Instantiate a PSF reconstructor object using GALACSI AO with
#NGS at 50 arcsec, pupil sampled at 128 pix and 'python' integrator

>>> r = rpsfpy.Reconstruct(pixdiam=128., ao_system="GALACSI", ngspos=[[50.,0.]], atmosphere=a,integrator='python')

#Objects positions in the field in arcsec

>>> objects = [[0.,0.], [0., 32.]]

#Lambda in nm

>>> lambdaim = 640

#compute PSF

>>> psf = r.psf(objects, lambdaim., out="psfs.fits")

#In this example outputs FITS file 'psf.fits'
