import numpy as np
import rpsfpy
import pyfits

def run():
    #Set up atmospheric parameters
    atmosphere = {'dr0':35., 'cn2': np.array([0.8, 0.2]), 'h_profile': np.array([0., 10000.]), 'L0': 25.}

    #diameter in pixels
    pixdiam = 24

    #pupil diameter in m
    pupdiam = 8.

    #Number of Zernike modes
    zmodes = 60

    #Sctructure class instance
    structure = rpsfpy.Structure(pixdiam, pupdiam, atmosphere, zmodes)

    #Object position in the field in arcsec
    objpos = [0.,0.]

    #NGS postion in arcsec
    ngspos = [45.,0.]

    #LGS positions in arcsec
    lgspos = np.array(([32.,-32.],[-32.,32.],[32.,32.],[-32.,-32.]))

    #LGS height in m
    lgs_height = 90000.

    #Lambda in nm
    lambdaim = 600

    #compute PSF
    psf = structure.psf(objpos, ngspos, lgspos, lgs_height, lambdaim)
if __name__ == "__main__":
    run()