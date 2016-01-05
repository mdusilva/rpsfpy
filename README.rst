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

>>> structure = rpsfpy.Structure() # can also give file as input
>>> structure = rpsfpy.Structure('foobar.cfg')

#Object position in the field in arcsec

>>> objpos = np.array(([[0.,0.]]))

#Lambda in nm

>>> lambdaim = 600

#compute PSF

>>> psf = structure.psf(objpos, lambdaim, out='psf.fits')


In order to input the system and atmospheric parameters edit the file "default.cfg"
