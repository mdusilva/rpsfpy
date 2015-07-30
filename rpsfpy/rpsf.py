import os

import numpy as np
from scipy.special import jn
from scipy.integrate import trapz, quad
from scipy import signal
import zernike
import pyfits

#zernike coefficients correlation mask
maskfile = os.path.join(os.path.dirname(__file__), r'masknn.fits')
z_handle = pyfits.open(maskfile)
zcoef_mask = z_handle[0].data

def _chassat_integral(x, zeta=None, Lam=None, w=None, n1=None, n2=None, m1=None, m2=None, z1=None, z2=None):
    """Inner integral to compute the correlations (Chassat)"""
    #compute bessel functions
    j1 = jn(n1+1,x)
    j2 = jn(n2+1,w*x)
    j3 = jn(m1+m2,zeta*x)
    j4 = jn(np.abs(m1-m2),zeta*x)

    #Compute K1 and K2
    if m1 == 0:
        if m2 == 0:
            K1 = 1.
        elif z2 % 2 == 0:
            K1 = (-1.)**m2 * np.sqrt(2.)
        else:
            K1 = 0.
    else:
        if z1 % 2 == 0:
            if m2 == 0:
                K1 = np.sqrt(2.)
            elif z2 % 2 == 0:
                K1 = (-1.)**m2
            else:
                K1 = 0.
        else:
            if m2 == 0:
                K1 = 0.
            elif z2 % 2 == 0:
                K1 = 0.
            else:
                K1 = (-1.)**(m2+1)
    if m1 == 0:
        K2 = 0.
    elif z1 % 2 ==0:
        if m2 != 0 and z2 % 2 == 0:
            if (m1 - m2) % 2 == 0:
                K2 = 1.
            else:
                if (m1 - m2) < 0:
                    K2 = -1.
                else:
                    K2 = 1.
        else:
            K2 = 0.                  
    else:
        if m2 != 0 and z2 % 2 != 0:
            if (m1 - m2) % 2 == 0:
                K2 = 1.
            else:
                if (m1 - m2) < 0:
                    K2 = -1.
                else:
                    K2 = 1.
        else:
            K2 = 0.
    I = x**(-14./3.) * j1 * j2 * (1. + (Lam / x)**2.)**(-11./6.) * (K1/w * j3 + K2/w * j4)
    return I

def fftcorr(X,Y,s=None):
    """Compute cross-correlation or autocorrelation using FFT."""
    if hasattr(X,'shape') and hasattr(Y,'shape'):
        if len(X.shape) == 2 and len(Y.shape) == 2:
            f1 = np.fft.fft2(X,s=s)
            f2 = np.fft.fft2(Y,s=s)
            return np.real(np.fft.ifft2(f1*np.conj(f2)))
        else:
            raise AttributeError('Input must be two 2d arrays')
    else:
        raise AttributeError('Input must be two 2d arrays')

def polar2(radius, oc=0., fourpixels=True, leq=False, length=None, center=None):

    """Compute pupil mask and polar coordinates.
    
    Assumes a circular pupil with some radius given in pixels and computes
    arrays containing grids of polar coordinates rho and phi and an array
    containing the pupil mask (i.e. 1 inside the pupil and 0 outside).
    
    Parameters
    ----------
    radius: int or float
            radius of pupil in pixels
    oc: float, optional
        size of central occultation for pupil mask (default=0.)
    fourpixels: bool
        if True and length is < 0 or missing then the centre of the pupil is
        in the intersection of the four central pixels and the length set to 
        an even number, if False the centre is in central pixel and length set to 
        an odd number (of pixels)
    leq: bool
        if True the border of mask is included (points <= radius), if False it 
        is not (points < radius)
    length: float, optional
        length of pupil in physical units, computed according to fourpixels argument
        if not given
    center: array_like, optional
        coordinates of pupil center, set to half of length if not given
        
    Returns
    -------
    
    rho: ndarray
        Array of rho coordinates
    phi: ndarray
        Array of phi coordinates
    mask: ndarray
        Array of pupil's mask (i.e. 1 inside the pupil and 0 outside)
        
    """
    if length <= 0. or length is None:
        if fourpixels:
            length = 2 * np.rint(radius)
        else:
            length = 2 * np.trunc(radius) + 1
    if center:
        cx = center[0]
        cy = center[1]
    else:
        cx = (length - 1.) / 2.
        cy = (length - 1.) / 2.
    y, x = np.mgrid[0:length,0:length]
    y = y - cy
    x = x - cx
    rho = np.sqrt(x**2. + y**2.) / radius
    phi = np.arctan2(y,x+(rho<=0.))
    mask = np.where(np.logical_and(rho <= 1., rho >= oc),1.,0.)
    if leq:
        mask = np.where(np.logical_and(rho <= 1., rho >= oc),1.,0.)
    else:
        mask = np.where(np.logical_and(rho < 1., rho >= oc),1.,0.)

    return rho, phi, mask

def angletiti(x, y):
    """Compute angle for a point x,y and convert to 0-360 degree range."""
    if x > 0. and y > 0.:
        gamma = np.arctan2(y,x)*180./np.pi
    if x < 0. and y > 0.:
        gamma = np.arctan2(y,x)*180./np.pi
    if x < 0. and y < 0.:
        gamma = 360.+np.arctan2(y,x)*180./np.pi
    if x > 0. and y < 0.:
        gamma = 360.+np.arctan2(y,x)*180./np.pi
    if x == 0. and y < 0.:
        gamma = 270. 
    if x == 0. and y > 0.:
        gamma = 90.
    if x > 0. and y == 0.:
        gamma = 0.
    if x < 0. and y == 0.:
        gamma = 180.
    if (np.abs(x) < 1.e-6) and (np.abs(y) < 1.e-6):
        gamma = 0.
    return gamma 

class Structure(object):
    """
    Muse GLAO structure functions
    Anisoplanetism and fitting errors only
    """
    def __init__(self, pixdiam, pupil_diameter, atmosphere, nzer=1):
        self.Zernike = zernike.Zernike()
        self.pixdiam = pixdiam  #must be integer
        self.cn2 = atmosphere['cn2']
        self.h_profile = atmosphere['h_profile']
        self.pupil_diameter = pupil_diameter
        self.dr0 = atmosphere['dr0']
        self.large_scale = atmosphere['L0']
        self.nz1 = nzer
        self.nz2 = nzer
        self.lgscoefmatrix = np.zeros((12, self.nz1-4, self.nz2-4)) #make as function of number of lgs
        self.ngscoefmatrix = np.zeros((2,2,2))
        self.propervectors = np.zeros((12, self.nz1-4, self.nz2-4))
        self.propervalues = np.zeros((12, self.nz1-4))
        self.rho, self.phi, self.mask = polar2(self.pixdiam/2., fourpixels=True, length=self.pixdiam)
        self.zernikes = np.zeros((self.nz1-2, self.pixdiam, self.pixdiam))
        self.lgsvmatrices = np.zeros((15, self.nz1-4, 2*self.pixdiam, 2*self.pixdiam))   #make as function of number of lgs
        self.ngsvmatrices = np.zeros((2, 2, 2*self.pixdiam, 2*self.pixdiam))
        self.lgsnewzernikes = np.zeros((15, self.nz1-4, self.pixdiam, self.pixdiam))
        self.ngsnewzernikes = np.zeros((2, 2, self.pixdiam, self.pixdiam))

    def correl_swsw(self, angle, nz1, nz2, h1, h2):
        """Correlation coeficients between two spherical waves."""
        alpha = angle * 4.85*1.e-6
        R = self.pupil_diameter / 2.
        R1 = (h1-self.h_profile) / h1 * R
        R2 = (h2-self.h_profile) / h2 * R
        zeta = alpha  * self.h_profile / R1
        w = R2 / R1
        Lam = 2. * np.pi * R1 / self.large_scale
        n1, m1 = zernike.noll2zern(nz1)
        n2, m2 = zernike.noll2zern(nz2)
        results = np.zeros(len(self.h_profile))
        for idx in np.arange(len(self.h_profile)):
            result_quad = quad(_chassat_integral, 0, np.inf, args=(zeta[idx], Lam[idx], w[idx], n1, n2, m1, m2, nz1, nz2))
            results[idx] = result_quad[0] * self.cn2[idx] * R1[idx]**(5./3.)
        if len(results) < 2:
            final_integral = results / (self.cn2 * R1**(5./3.))
        else:
            final_integral = trapz(results, x=self.h_profile) / trapz(self.cn2 * R1**(5./3.), x=self.h_profile)
        final_integral = 3.895 * (-1.)**((n1+n2-m1-m2)/2.) * np.sqrt((n1+1.)*(n2+1.)) * self.dr0**(5./3.) * final_integral    
        return final_integral

    def diagcoef(self, M, type='svd'):
        """Diagonalize coeficient matrices, returns eigen values and vectors."""
        #with svd
        if type == 'svd':
            U, s, V = np.linalg.svd(M, full_matrices=False)
            U = U.T
        #with eigen vectors
        if type == 'eig':
            s, U = np.linalg.eig(M)
            U = U.T
        return s, U

    def compvii(self, z, pupil):
        """Computation of the Vii correlation matrix."""
        nz = len(z)
        imshape = (2*z.shape[1], 2*z.shape[2])
        corrPP = np.rint(np.abs(fftcorr(pupil,pupil,s=imshape)))
        mask = (corrPP>0.).astype(int)
        corrPP = np.where(corrPP>1.,corrPP,1.)
        u = np.zeros((nz, 2*z.shape[1], 2*z.shape[2]))
        corrPP = 1. / corrPP
        for i in range(nz):
            corij = fftcorr(z[i,:,:], z[i,:,:], s=imshape)
            corzw = fftcorr(z[i,:,:] * z[i,:,:], pupil, s=imshape)
            u[i,:,:] = np.real(2. * (corzw - corij)) * mask * np.real(corrPP)
            u[i,:,:] = np.fft.fftshift(u[i,:,:])
        return u

    def Dfitting(self, lambdaim, fc_constant=0.37):
        """Fitting error structure function."""
        rho, phi, mask = polar2(self.pixdiam, length=2*self.pixdiam, center=[self.pixdiam,self.pixdiam])
        radial, azi = zernike.noll2zern(self.nz1)
        tot_correletion_aiaj = np.zeros((2*self.pixdiam,2*self.pixdiam))
        r0 = 1. / self.dr0
        Fc = fc_constant * (radial + 1.)
        for i in range(2*self.pixdiam):
            for j in range(2*self.pixdiam):
                lower_bound = 2. * np.pi * Fc * rho[i,j]
                result_quad = quad(lambda x: x**(-8./3.)*(1.-jn(0,x)), lower_bound, 150.)
                tot_correletion_aiaj[i,j] = result_quad[0] * (rho[i,j]/r0)**(5./3.)
        return 0.023 * 2.**(11./3.) * np.pi**(8./3.) * tot_correletion_aiaj
        
    def Dngs(self, ngspos, objectpos, hngs=10.e20):
        """Natural guide star anisoplanetism structure function"""
#        hngs = 10.e20 #try later with infinite
        xngs, yngs = ngspos
        xob, yob = objectpos
        alphangs = np.sqrt((xngs-xob)**2.+(yngs-yob)**2.)
        gammangs = angletiti(xngs-xob, yngs-yob)
        for i in np.arange(2,4):
            for j in np.arange(2,4):
                #aNGaNG
                self.ngscoefmatrix[0,i-2,j-2] = self.correl_swsw(0., i, j, hngs, hngs)
                #aiaNG
                self.ngscoefmatrix[1,i-2,j-2] = self.correl_swsw(alphangs,i, j, hngs, hngs)
        
        evalues0, evectors0 = self.diagcoef(self.ngscoefmatrix[0])
        evalues1, evectors1 = self.diagcoef(self.ngscoefmatrix[1])
        self.zernikes[0] = self.Zernike.rotate(self.mask*self.rho, self.phi, 2, -gammangs)
        self.zernikes[1] = self.Zernike.rotate(self.mask*self.rho, self.phi, 3, -gammangs)
        self.ngsnewzernikes[0] = np.dot(evectors0, self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
        self.ngsnewzernikes[1] = np.dot(evectors1, self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
        self.ngsvmatrices[0] = self.compvii(self.ngsnewzernikes[0], self.mask)
        self.ngsvmatrices[1] = self.compvii(self.ngsnewzernikes[1], self.mask)
        dphings = 2. * np.tensordot(evalues0, self.ngsvmatrices[0], axes=1) - 2. *  np.tensordot(evalues1, self.ngsvmatrices[1], axes=1)
        return dphings

    def Dlgs(self, lgspos, hlgs, objectpos, hngs = 10.e20):
        """Laser guide stars anisoplanetism structure function"""
        nlgs = len(lgspos)
#        hngs = 10.e20 #try later with infinite
        self.sigmaslgs = np.ones(nlgs) / nlgs
        xlgs, ylgs = lgspos.T
        xob, yob = objectpos
        alphalgs = np.zeros((nlgs,nlgs))
        gammalgs = np.zeros(nlgs)
        betalgs = np.zeros(6)
        betalgs[0] = angletiti(xlgs[1]-xlgs[0], ylgs[1]-ylgs[0])
        betalgs[1] = angletiti(xlgs[2]-xlgs[0], ylgs[2]-ylgs[0])
        betalgs[2] = angletiti(xlgs[3]-xlgs[0], ylgs[3]-ylgs[0])
        betalgs[3] = angletiti(xlgs[2]-xlgs[1], ylgs[2]-ylgs[1])
        betalgs[4] = angletiti(xlgs[3]-xlgs[1], ylgs[3]-ylgs[1])
        betalgs[5] = angletiti(xlgs[3]-xlgs[2], ylgs[3]-ylgs[2])
        for i in np.arange(nlgs):
            gammalgs[i] = angletiti(xlgs[i]-xob, ylgs[i]-yob)
            for j in np.arange(i,nlgs):
                if i == j:
                    alphalgs[i,j] = np.sqrt((xlgs[i]-xob)**2. + (ylgs[i]-yob)**2.)
                else:
                    alphalgs[i,j] = np.sqrt((xlgs[i]-xlgs[j])**2. + (ylgs[i]-ylgs[j])**2.)
                    alphalgs[j,i] = alphalgs[i,j]
        for i in np.arange(4,self.nz1):
            for j in np.arange(i,self.nz2):
                if zcoef_mask[j-2,i-2]:
                    #aiaj
                    self.lgscoefmatrix[0,i-4,j-4] = self.correl_swsw(0.,i, j, hngs, hngs)
                    #algsalgs
                    self.lgscoefmatrix[1,i-4,j-4] = self.correl_swsw(0.,i, j, hlgs, hlgs)
                    #aialgs1
                    self.lgscoefmatrix[2,i-4,j-4] = self.correl_swsw(alphalgs[0,0],i, j, hngs, hlgs)
                    #aialgs2
                    self.lgscoefmatrix[3,i-4,j-4] = self.correl_swsw(alphalgs[1,1],i, j, hngs, hlgs)
                    #aialgs3
                    self.lgscoefmatrix[4,i-4,j-4] = self.correl_swsw(alphalgs[2,2],i, j, hngs, hlgs)
                    #aialgs4
                    self.lgscoefmatrix[5,i-4,j-4] = self.correl_swsw(alphalgs[3,3],i, j, hngs, hlgs)
                    #algs1algs2
                    self.lgscoefmatrix[6,i-4,j-4] = self.correl_swsw(alphalgs[0,1],i, j, hlgs, hlgs)
                    #algs1algs4
                    self.lgscoefmatrix[7,i-4,j-4] = self.correl_swsw(alphalgs[0,3],i, j, hlgs, hlgs)
                    #algs2algs3
                    self.lgscoefmatrix[8,i-4,j-4] = self.correl_swsw(alphalgs[1,2],i, j, hlgs, hlgs)
                    #algs3algs4
                    self.lgscoefmatrix[9,i-4,j-4] = self.correl_swsw(alphalgs[2,3],i, j, hlgs, hlgs)
                    #algs2algs4
                    self.lgscoefmatrix[10,i-4,j-4] = self.correl_swsw(alphalgs[1,3],i, j, hlgs, hlgs)
                    #algs1algs3
                    self.lgscoefmatrix[11,i-4,j-4] = self.correl_swsw(alphalgs[0,2],i, j, hlgs, hlgs)
                if j != i:
                    for k in range(12):
                        self.lgscoefmatrix[k,j-4,i-4] = self.lgscoefmatrix[k,i-4,j-4]
        for i in np.arange(12):
            self.propervalues[i], self.propervectors[i] = self.diagcoef(self.lgscoefmatrix[i])
        for p in np.arange(nlgs):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammalgs[p])            
            self.lgsnewzernikes[p+1,:,:,:] = np.dot(self.propervectors[1],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            self.lgsvmatrices[p+1] = self.compvii(self.lgsnewzernikes[p+1], self.mask)
        for p in np.arange(nlgs):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammalgs[p])            
            self.lgsnewzernikes[p+1+nlgs,:,:,:] = np.dot(self.propervectors[p+2],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            self.lgsvmatrices[p+1+nlgs] = self.compvii(self.lgsnewzernikes[p+1+nlgs], self.mask)
        for p in np.arange(6):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -betalgs[p])
            self.lgsnewzernikes[p+9,:,:,:] = np.dot(self.propervectors[p+6],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            self.lgsvmatrices[p+9] = self.compvii(self.lgsnewzernikes[p+9], self.mask)
        for i in np.arange(4,self.nz2):
            self.zernikes[i-2] = self.Zernike.zernike(self.mask*self.rho, self.phi, i)
        self.lgsnewzernikes[0,:,:,:] = np.dot(self.propervectors[0],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
        self.lgsvmatrices[0] = self.compvii(self.lgsnewzernikes[0], self.mask)       
        dphilgs = np.tensordot(self.propervalues[0],self.lgsvmatrices[0],axes=1) - 2. * np.tensordot(self.propervalues[2:6] * self.sigmaslgs.reshape((-1,1)), self.lgsvmatrices[5:9], axes=([0,1],[0,1])) + \
                 self.sigmaslgs[0]**2. * np.tensordot(self.propervalues[1], self.lgsvmatrices[1], axes=1) + self.sigmaslgs[1]**2. * np.tensordot(self.propervalues[1], self.lgsvmatrices[2], axes=1) + \
                 self.sigmaslgs[2]**2. * np.tensordot(self.propervalues[1], self.lgsvmatrices[3], axes=1) + self.sigmaslgs[3]**2. * np.tensordot(self.propervalues[1], self.lgsvmatrices[4], axes=1) + \
                 2. * self.sigmaslgs[0]*self.sigmaslgs[1] * np.tensordot(self.propervalues[6], self.lgsvmatrices[9], axes=1) + 2. * self.sigmaslgs[0]*self.sigmaslgs[3] * np.tensordot(self.propervalues[7], self.lgsvmatrices[10], axes=1) + \
                 2. * self.sigmaslgs[1]*self.sigmaslgs[2] * np.tensordot(self.propervalues[8], self.lgsvmatrices[11], axes=1) + 2. * self.sigmaslgs[2]*self.sigmaslgs[3] * np.tensordot(self.propervalues[9], self.lgsvmatrices[12], axes=1) + \
                 2. * self.sigmaslgs[1]*self.sigmaslgs[3] * np.tensordot(self.propervalues[10], self.lgsvmatrices[13], axes=1) + 2. * self.sigmaslgs[0]*self.sigmaslgs[2] * np.tensordot(self.propervalues[11], self.lgsvmatrices[14], axes=1)
        return dphilgs

    def otf(self, objectlist, ngspos, lgspos, hlgs, lambdaim, out=None, **kwargs):
        """Compute OTF from Structrure functions"""
        otf_array = []
        for objectpos in objectlist:
            if  "hngs" in kwargs:
                dphings = self.Dngs(ngspos, objectpos, hngs=kwargs[hngs])
                dphilgs = self.Dlgs(lgspos, hlgs, objectpos, hngs=kwargs[hngs])
            else:
                dphings = self.Dngs(ngspos, objectpos)
                dphilgs = self.Dlgs(lgspos, hlgs, objectpos)
            if "fc_constant" in kwargs:
                dfitting = self.Dfitting(lambdaim, fc_constant=kwargs[fc_constant])
            else:
                dfitting = self.Dfitting(lambdaim)
            dphi_tot = dphings + dphilgs + dfitting
            ac = fftcorr(self.mask,self.mask,s=(2*self.pixdiam, 2*self.pixdiam))
            ac = np.fft.fftshift(ac)
            ao = np.exp(-dphi_tot/2.)
            otf = ac * ao
            otf_array.append(otf)
        otf_array = np.array(otf_array)
        if out:
            try:
                filename, fileExtension = os.path.splitext(out)
                if fileExtension == ".fits" or fileExtension == ".FITS":
                    hdu = pyfits.PrimaryHDU(otf_array)
                    hdu.writeto(out, clobber=True)
                else:
                    raise IOError("File must be of FITS format")
            except IOError:
                print "An error ocurred while writing to file "+str(out)
        return otf_array

    def psf(self, objectlist, ngspos, lgspos, hlgs, lambdaim, out=None, otfs=None, **kwargs):
        """Compute PSF from Structrure functions"""
        psf_array = []
        otf_array = self.otf(objectlist, ngspos, lgspos, hlgs, lambdaim, out=otfs, **kwargs)
        for otf in otf_array:
            psf = np.abs(np.fft.fftshift(np.fft.ifft2(otf)))
            psf = psf / np.sum(psf)
            psf_array.append(psf)
        psf_array = np.array(psf_array)
        if out:
            try:
                filename, fileExtension = os.path.splitext(out)
                if fileExtension == ".fits" or fileExtension == ".FITS":
                    hdu = pyfits.PrimaryHDU(psf_array)
                    try:
                        hdu.header['scale'] = (self._scale(lambdaim), "arcsec per pixel")
                    except KeyError:
                        hdu.header.update(key="pscale", value=self._pscale(lambdaim), comment="arcsec per pixel")
                    hdu.writeto(out, clobber=True)
                else:
                    raise IOError("File must be of FITS format")
            except IOError:
                print "An error ocurred while writing to file "+str(out)
        return psf_array

    def _scale(self,lambdaim):
        """Compute plate scale in arc sec per pixel, lamda in nm"""
        theta = lambdaim * 1.e-9 / (2. * self.pupil_diameter)
        return 206264.8 * theta / self.pixdiam