rpsfpy: GLAO PSF reconstruction in Python

This is a set of tools to reconstruct PSFs of GLAO systems, taking 
into account the anisoplanetism effect.

Contributors: M. Silva, T. Fusco, R. Villecroze, B. Bneichel, R Bacon,
P-Y Madec, M. Lelouran, A. Guesalaga

-------------------------
Usage example:

>>> import rpsfpy

#Set up atmospheric parameters

>>> atmosphere = {'dr0':32.3, 'cn2': np.array([1.,]), 'h_profile': np.array([200.,]), 'L0': 25.}

#diameter in pixels

>>> pixdiam = 128

#pupil diameter in m

>>> pupdiam = 8.

#Number of Zernike modes

>>> zmodes = 980

#Sctructure class instance

>>> structure = rpsfpy.Structure(pixdiam, pupdiam, atmosphere, zmodes)

#Object position in the field in arcsec

>>> objpos = np.array(([0.,0.],[6.9,0.],[0.,6.9],[-6.9,0.],[0.,-6.9]))

#NGS postion in arcsec

>>> ngspos = [-75.6,-75.6]

#LGS positions in arcsec

>>> lgspos = np.array(([64.1,0.],[0.,64.1],[-64.1,0.],[0.,-64.1]))

#LGS height in m

>>> lgs_height = 90000.

#Lambda in nm

>>> lambdaim = 600

#compute PSF

>>> psf = structure.psf(objpos, ngspos, lgspos, lgs_height, lambdaim, out='psf.fits')
