import os

import numpy as np
from scipy.special import jv
from scipy.integrate import trapz, quad, romberg
from scipy import signal
import itertools
import zernike
import pyfits
import parameters

import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count


#zernike coefficients correlation mask
maskfile = os.path.join(os.path.dirname(__file__), r'masknn.fits')
z_handle = pyfits.open(maskfile)
zcoef_mask = z_handle[0].data

def compmask(nz1, nz2):
    first = 2
    masku = np.zeros((nz1,nz2))
    for i in range(nz1):
        for j in range(i, nz2):
            masku[i,j] = (zernike.noll2zern(i+first)[1] % 2) == (zernike.noll2zern(j+first)[1] % 2)
    return masku.astype(int)

#def _chassat_integral(x, zeta=None, Lam=None, w=None, n1=None, n2=None, m1=None, m2=None, z1=None, z2=None):
#    """Inner integral to compute the correlations (Chassat)"""
#    #compute bessel functions
#    j1 = jn(n1+1,x)
#    j2 = jn(n2+1,w*x)
#    j3 = jn(m1+m2,zeta*x)
#    j4 = jn(np.abs(m1-m2),zeta*x)
#
#    #Compute K1 and K2
#    if m1 == 0:
#        if m2 == 0:
#            K1 = 1.
#        elif z2 % 2 == 0:
#            K1 = (-1.)**m2 * np.sqrt(2.)
#        else:
#            K1 = 0.
#    else:
#        if z1 % 2 == 0:
#            if m2 == 0:
#                K1 = np.sqrt(2.)
#            elif z2 % 2 == 0:
#                K1 = (-1.)**m2
#            else:
#                K1 = 0.
#        else:
#            if m2 == 0:
#                K1 = 0.
#            elif z2 % 2 == 0:
#                K1 = 0.
#            else:
#                K1 = (-1.)**(m2+1)
#    if m1 == 0:
#        K2 = 0.
#    elif z1 % 2 ==0:
#        if m2 != 0 and z2 % 2 == 0:
#            if (m1 - m2) % 2 == 0:
#                K2 = 1.
#            else:
#                if (m1 - m2) < 0:
#                    K2 = -1.
#                else:
#                    K2 = 1.
#        else:
#            K2 = 0.                  
#    else:
#        if m2 != 0 and z2 % 2 != 0:
#            if (m1 - m2) % 2 == 0:
#                K2 = 1.
#            else:
#                if (m1 - m2) < 0:
#                    K2 = -1.
#                else:
#                    K2 = 1.
#        else:
#            K2 = 0.
#    I = x**(-14./3.) * j1 * j2 * (1. + (Lam / x)**2.)**(-11./6.) * (K1/w * j3 + K2/w * j4)
#    return I

def _kvalues(n1=None, n2=None, m1=None, m2=None, z1=None, z2=None):
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
    return K1, K2

def _chassat_integral(x, zeta=None, Lam=None, w=None, K1=None, K2=None, n1=None, n2=None, m1=None, m2=None):
    """Inner integral to compute the correlations (Chassat)"""
    #compute bessel functions
    j1 = jv(n1+1,x)
    j2 = jv(n2+1,w*x)
    j3 = jv(m1+m2,zeta*x)
    j4 = jv(np.abs(m1-m2),zeta*x)
    return x**(-14./3.) * j1 * j2 * (1. + (Lam / x)**2.)**(-11./6.) * (K1/w * j3 + K2/w * j4)

def _modified_chassat(t, zeta=None, Lam=None, w=None, K1=None, K2=None, n1=None, n2=None, m1=None, m2=None):
    return _chassat_integral(np.tan(t), zeta, Lam, w, K1, K2, n1, n2, m1, m2) / np.cos(t)**2.

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
    def __init__(self, cfile='default.cfg'):
        if cfile == 'default.cfg':
            configuration_file = os.path.join(os.path.dirname(__file__), cfile)
        else:
            configuration_file = cfile
        atmosphere, zm, D, pixels, ngs_array, lgs_array, lgs_height =  parameters.read(configuration_file)
        self.Zernike = zernike.Zernike()
        self.pixdiam = pixels  #must be integer
        self.cn2 = atmosphere['cn2']
        self.h_profile = atmosphere['h_profile']
        self.pupil_diameter = D
        self.dr0 = atmosphere['dr0']
        self.large_scale = atmosphere['L0']
        self.nz1 = zm
        self.nz2 = zm
        self.ngspos = ngs_array
        self.lgspos = lgs_array
        self.hlgs = lgs_height
#        self.lgscoefmatrix = np.zeros((12, self.nz1-4, self.nz2-4)) #make as function of number of lgs
#        self.ngscoefmatrix = np.zeros((2,2,2))
#        self.propervectors = np.zeros((12, self.nz1-4, self.nz2-4))
#        self.propervalues = np.zeros((12, self.nz1-4))
        self.rho, self.phi, self.mask = polar2(self.pixdiam/2., fourpixels=True, length=self.pixdiam)
        self.zernikes = np.zeros((self.nz1-2, self.pixdiam, self.pixdiam))
#        self.lgsvmatrices = np.zeros((15, self.nz1-4, 2*self.pixdiam, 2*self.pixdiam))   #make as function of number of lgs
        self.ngsvmatrices = np.zeros((2, 2, 2*self.pixdiam, 2*self.pixdiam))
#        self.lgsnewzernikes = np.zeros((15, self.nz1-4, self.pixdiam, self.pixdiam))
        self.ngsnewzernikes = np.zeros((2, 2, self.pixdiam, self.pixdiam))
        if self.nz1 > 980:
            self.zcoef_mask = compmask(self.nz1, self.nz2).T
        else:
            self.zcoef_mask = zcoef_mask

    def correl_swsw(self, angle, nz1, nz2, h1, h2, method='quad'):
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
        k1, k2 = _kvalues(n1, n2, m1, m2, nz1, nz2)
        results = np.zeros(len(self.h_profile))
        for idx in np.arange(len(self.h_profile)):
            if method == 'quad':
                result_quad = quad(_chassat_integral, 0, np.inf, args=(zeta[idx], Lam[idx], w[idx],  k1, k2, n1, n2, m1, m2))
                results[idx] = result_quad[0] * self.cn2[idx] * R1[idx]**(5./3.)
            elif method == 'romberg':
                result_quad = romberg(_modified_chassat, 1.e-26,np.pi/2., args=(zeta[idx], Lam[idx], w[idx], k1, k2, n1, n2, m1, m2), vec_func = False)
                results[idx] = result_quad * self.cn2[idx] * R1[idx]**(5./3.)
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
                result_quad = quad(lambda x: x**(-8./3.)*(1.-jv(0,x)), lower_bound, 150.)
                tot_correletion_aiaj[i,j] = result_quad[0] * (rho[i,j]/r0)**(5./3.)
        return 0.023 * 2.**(11./3.) * np.pi**(8./3.) * tot_correletion_aiaj
        
    def Dngs(self, objectpos, hngs=10.e20):
        """Natural guide star anisoplanetism structure function"""
#        hngs = 10.e20 #try later with infinite
        nngs = len(self.ngspos)
#        hngs = 10.e20 #try later with infinite
        self.sigmasngs = np.ones(nngs) / nngs
        xngs, yngs = self.ngspos.T
        xob, yob = objectpos
        alphangs = np.zeros((nngs,nngs))
        gammangs = np.zeros(nngs)
        combings = np.fromiter(itertools.combinations(np.arange(nngs),2), np.dtype(('i,i')))
        betangs = np.zeros(len(combings))
        for b_idx in np.arange(len(combings)):
            l_idx = combings[b_idx][1]
            r_idx = combings[b_idx][0]
            betangs[b_idx] = angletiti(xngs[l_idx]-xngs[r_idx], yngs[l_idx]-yngs[r_idx])
        
        for i in np.arange(nngs):
            gammangs[i] = angletiti(xngs[i]-xob, yngs[i]-yob)
            for j in np.arange(i,nngs):
                if i == j:
                    alphangs[i,j] = np.sqrt((xngs[i]-xob)**2. + (yngs[i]-yob)**2.)
                else:
                    alphangs[i,j] = np.sqrt((xngs[i]-xngs[j])**2. + (yngs[i]-yngs[j])**2.)
                    alphangs[j,i] = alphangs[i,j]
        ngscoefmatrix = np.zeros((2 + nngs + len(combings), 2, 2))
        print "computing NGS structure function"
        for i in np.arange(2,4):
            for j in np.arange(i,4):
                ngscoefmatrix[0,i-2,j-2] = self.correl_swsw(0.,i, j, hngs, hngs)
                ngscoefmatrix[1,i-2,j-2] = ngscoefmatrix[0,i-2,j-2]
                for i_ngs in np.arange(nngs):
                    ngscoefmatrix[2+i_ngs,i-2,j-2] = self.correl_swsw(alphangs[i_ngs,i_ngs],i, j, hngs, hngs)
                for i_ngs in np.arange(len(combings)):
                    l_idx = combings[i_ngs][1]
                    r_idx = combings[i_ngs][0]
                    ngscoefmatrix[2+nngs+i_ngs,i-2,j-2] = self.correl_swsw(alphangs[l_idx,r_idx],i, j, hngs, hngs)
                if j != i:
                    for k in range(2 + nngs + len(combings)):
                        ngscoefmatrix[k,j-2,i-2] = ngscoefmatrix[k,i-2,j-2]
        propervectors = np.zeros((2 + nngs + len(combings), 2, 2))
        propervalues = np.zeros((2 + nngs + len(combings), 2))
        ngsvmatrices = np.zeros((1+nngs+nngs+len(combings), 2, 2*self.pixdiam, 2*self.pixdiam))   #make as function of number of lgs
        ngsnewzernikes = np.zeros((1+nngs+nngs+len(combings), 2, self.pixdiam, self.pixdiam))
        for i in np.arange(2 + nngs + len(combings)):
            propervalues[i], propervectors[i] = self.diagcoef(ngscoefmatrix[i])
        for p in np.arange(nngs):
            for i in np.arange(2,4):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammangs[p])            
            ngsnewzernikes[p+1,:,:,:] = np.dot(propervectors[1],  self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
            ngsvmatrices[p+1] = self.compvii(ngsnewzernikes[p+1], self.mask)
        for p in np.arange(nngs):
            for i in np.arange(2,4):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammangs[p])            
            ngsnewzernikes[p+1+nngs,:,:,:] = np.dot(propervectors[p+2],  self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
            ngsvmatrices[p+1+nngs] = self.compvii(ngsnewzernikes[p+1+nngs], self.mask)
        for p in np.arange(len(combings)):
            for i in np.arange(2,4):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -betangs[p])
            ngsnewzernikes[p+1+nngs+nngs,:,:,:] = np.dot(propervectors[p+2+nngs],  self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
            ngsvmatrices[p+1+nngs+nngs] = self.compvii(ngsnewzernikes[p+1+nngs+nngs], self.mask)
        for i in np.arange(2,4):
            self.zernikes[i-2] = self.Zernike.zernike(self.mask*self.rho, self.phi, i)
        ngsnewzernikes[0,:,:,:] = np.dot(propervectors[0],  self.zernikes[0:2].reshape((2,self.pixdiam*self.pixdiam))).reshape((2,self.pixdiam,self.pixdiam))
        ngsvmatrices[0] = self.compvii(ngsnewzernikes[0], self.mask)
        term1 = np.tensordot(propervalues[0],ngsvmatrices[0],axes=1)
        term2 = 2. * np.tensordot(propervalues[2:2 + nngs] * self.sigmasngs.reshape((-1,1)), ngsvmatrices[1+nngs:1+nngs+nngs], axes=([0,1],[0,1]))
        term3 = 0.
        for idx_ngs in np.arange(nngs):
            term3 = term3 + self.sigmasngs[idx_ngs]**2. * np.tensordot(propervalues[1], ngsvmatrices[idx_ngs+1], axes=1)
        term4 = 0.
        for b_idx in np.arange(len(combings)):
            l_idx = combings[b_idx][1]
            r_idx = combings[b_idx][0]
            term4 = term4 + 2. * self.sigmasngs[l_idx]*self.sigmasngs[r_idx] * np.tensordot(propervalues[2 + nngs + b_idx], ngsvmatrices[1+nngs+nngs+b_idx], axes=1)
        dphings = term1 - term2 + term3 + term4
#        np.savetxt("dphings.dat", dphings)
        return dphings

    def _compute_lgscoef(self, modes, nlgs, hngs, hlgs, combilgs, alphalgs):
        coefmatrix = np.zeros((2 + nlgs + len(combilgs),len(modes)))
        for l in xrange(len(modes)):
            i = modes[l][0]
            j = modes[l][1]
            coefmatrix[0,l] = self.correl_swsw(0.,i, j, hngs, hngs)
            coefmatrix[1,l] = self.correl_swsw(0.,i, j, hlgs, hlgs)
            for i_lgs in xrange(nlgs):
                coefmatrix[2+i_lgs,l] = self.correl_swsw(alphalgs[i_lgs,i_lgs],i, j, hngs, hlgs)
            for i_lgs in xrange(len(combilgs)):
                coefmatrix[2+nlgs+i_lgs,l] = self.correl_swsw(alphalgs[combilgs[i_lgs][1],combilgs[i_lgs][0]],i, j, hlgs, hlgs)

#            if j != i:
#                for k in range(2 + nlgs + len(combilgs)):
#                    coefmatrix[k,j-4,i-4] = coefmatrix[k,i-4,j-4]
        return coefmatrix, modes

    def Dlgs(self, objectpos, hngs = 10.e20, parallel='auto'):
        """Laser guide stars anisoplanetism structure function"""
        nlgs = len(self.lgspos)
#        hngs = 10.e20 #try later with infinite
        self.sigmaslgs = np.ones(nlgs) / nlgs
        xlgs, ylgs = self.lgspos.T
        xob, yob = objectpos
        alphalgs = np.zeros((nlgs,nlgs))
        gammalgs = np.zeros(nlgs)
        combilgs = np.fromiter(itertools.combinations(np.arange(nlgs),2), np.dtype(('i,i')))
        betalgs = np.zeros(len(combilgs))
        for b_idx in np.arange(len(combilgs)):
            l_idx = combilgs[b_idx][1]
            r_idx = combilgs[b_idx][0]
            betalgs[b_idx] = angletiti(xlgs[l_idx]-xlgs[r_idx], ylgs[l_idx]-ylgs[r_idx])
        for i in np.arange(nlgs):
            gammalgs[i] = angletiti(xlgs[i]-xob, ylgs[i]-yob)
            for j in np.arange(i,nlgs):
                if i == j:
                    alphalgs[i,j] = np.sqrt((xlgs[i]-xob)**2. + (ylgs[i]-yob)**2.)
                else:
                    alphalgs[i,j] = np.sqrt((xlgs[i]-xlgs[j])**2. + (ylgs[i]-ylgs[j])**2.)
                    alphalgs[j,i] = alphalgs[i,j]
        lgscoefmatrix = np.zeros((2 + nlgs + len(combilgs), self.nz1-4, self.nz2-4))

        if parallel=='auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus>1:
            print "computing LGS structure function in parallel with "+str(cpus)+" CPUs"
        else:
            print "computing LGS structure function"
        modes = []
        for i in np.arange(4,self.nz1):
            for j in np.arange(i,self.nz2):
                if self.zcoef_mask[j-2,i-2]:
                    modes.append((i,j))
        
        modes = np.array(modes)

#        for i in np.arange(4,self.nz1):
#            for j in np.arange(i,self.nz2):
#                if zcoef_mask[j-2,i-2]:
#                    lgscoefmatrix[0,i-4,j-4] = self.correl_swsw(0.,i, j, hngs, hngs)
#                    lgscoefmatrix[1,i-4,j-4] = self.correl_swsw(0.,i, j, hlgs, hlgs)
#                    for i_lgs in np.arange(nlgs):
#                        lgscoefmatrix[2+i_lgs,i-4,j-4] = self.correl_swsw(alphalgs[i_lgs,i_lgs],i, j, hngs, hlgs)
#                    for i_lgs in np.arange(len(combilgs)):
#                        l_idx = combilgs[i_lgs][1]
#                        r_idx = combilgs[i_lgs][0]
#                        lgscoefmatrix[2+nlgs+i_lgs,i-4,j-4] = self.correl_swsw(alphalgs[l_idx,r_idx],i, j, hlgs, hlgs)
#                if j != i:
#                    for k in range(2 + nlgs + len(combilgs)):
#                        lgscoefmatrix[k,j-4,i-4] = lgscoefmatrix[k,i-4,j-4]

        if cpus>1:
            pool = Pool(processes=cpus)
            start = 0
            end = len(modes)
            step = (end-start)/cpus + 1            

            results=[]
            for c in xrange(cpus):
                start_i = start + c*step
                end_i = min(start+(c+1)*step, end)
                modes_split = modes[start_i:end_i]
                results.append(pool.apply_async(self._compute_lgscoef,args=(modes_split,nlgs,hngs,self.hlgs,combilgs,alphalgs)))
            pool.close()
            pool.join()
            for c in xrange(cpus):
                values_received, modes_received = results[c].get()
                for l in xrange(len(modes_received)):
                    i = modes_received[l][0]-4
                    j = modes_received[l][1]-4
                    lgscoefmatrix[:,i,j] = values_received[:,l]
        else:
            compute_lgscoef_temp, received_modes = self._compute_lgscoef(modes,nlgs,hngs,self.hlgs,combilgs,alphalgs)
            for l in xrange(len(modes)):
                    i = modes[l][0]-4
                    j = modes[l][1]-4
                    lgscoefmatrix[:,i,j] = compute_lgscoef_temp[:,l]

        for i in np.arange(4,self.nz1):
            for j in np.arange(i,self.nz2):
                if j != i:
                    for k in range(2 + nlgs + len(combilgs)):
                        lgscoefmatrix[k,j-4,i-4] = lgscoefmatrix[k,i-4,j-4]

        propervectors = np.zeros((2 + nlgs + len(combilgs), self.nz1-4, self.nz2-4))
        propervalues = np.zeros((2 + nlgs + len(combilgs), self.nz1-4))
        lgsvmatrices = np.zeros((1+nlgs+nlgs+len(combilgs), self.nz1-4, 2*self.pixdiam, 2*self.pixdiam))   #make as function of number of lgs
        lgsnewzernikes = np.zeros((1+nlgs+nlgs+len(combilgs), self.nz1-4, self.pixdiam, self.pixdiam))
        for i in np.arange(2 + nlgs + len(combilgs)):
            propervalues[i], propervectors[i] = self.diagcoef(lgscoefmatrix[i])
        for p in np.arange(nlgs):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammalgs[p])            
            lgsnewzernikes[p+1,:,:,:] = np.dot(propervectors[1],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            lgsvmatrices[p+1] = self.compvii(lgsnewzernikes[p+1], self.mask)
        for p in np.arange(nlgs):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -gammalgs[p])            
            lgsnewzernikes[p+1+nlgs,:,:,:] = np.dot(propervectors[p+2],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            lgsvmatrices[p+1+nlgs] = self.compvii(lgsnewzernikes[p+1+nlgs], self.mask)
        for p in np.arange(len(combilgs)):
            for i in np.arange(4,self.nz2):
                self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, -betalgs[p])
            lgsnewzernikes[p+1+nlgs+nlgs,:,:,:] = np.dot(propervectors[p+2+nlgs],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
            lgsvmatrices[p+1+nlgs+nlgs] = self.compvii(lgsnewzernikes[p+1+nlgs+nlgs], self.mask)
        for i in np.arange(4,self.nz2):
            self.zernikes[i-2] = self.Zernike.zernike(self.mask*self.rho, self.phi, i)
        lgsnewzernikes[0,:,:,:] = np.dot(propervectors[0],  self.zernikes[2:].reshape((self.nz2-4,self.pixdiam*self.pixdiam))).reshape((self.nz2-4,self.pixdiam,self.pixdiam))
        lgsvmatrices[0] = self.compvii(lgsnewzernikes[0], self.mask)
        term1 = np.tensordot(propervalues[0],lgsvmatrices[0],axes=1)
        term2 = 2. * np.tensordot(propervalues[2:2 + nlgs] * self.sigmaslgs.reshape((-1,1)), lgsvmatrices[1+nlgs:1+nlgs+nlgs], axes=([0,1],[0,1]))
        term3 = 0.
        for idx_lgs in np.arange(nlgs):
            term3 = term3 + self.sigmaslgs[idx_lgs]**2. * np.tensordot(propervalues[1], lgsvmatrices[idx_lgs+1], axes=1)
        term4 = 0.
        for b_idx in np.arange(len(combilgs)):
            l_idx = combilgs[b_idx][1]
            r_idx = combilgs[b_idx][0]
            term4 = term4 + 2. * self.sigmaslgs[l_idx]*self.sigmaslgs[r_idx] * np.tensordot(propervalues[2 + nlgs + b_idx], lgsvmatrices[1+nlgs+nlgs+b_idx], axes=1)
        dphilgs = term1 - term2 + term3 + term4
        return dphilgs

    def otf(self, objectlist, lambdaim, out=None, parallel='auto', **kwargs):
        """Compute OTF from Structrure functions"""
        otf_array = []
        for objectpos in objectlist:
            if  "hngs" in kwargs:
                dphings = self.Dngs(objectpos, hngs=kwargs[hngs])
                dphilgs = self.Dlgs(objectpos, parallel=parallel, hngs=kwargs[hngs])
            else:
                dphings = self.Dngs(objectpos)
                dphilgs = self.Dlgs(objectpos, parallel=parallel)
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

    def psf(self, objectlist, lambdaim, out=None, otfs=None, parallel='auto', **kwargs):
        """Compute PSF from Structrure functions"""
        psf_array = []
#        print "Objects coordinates = ", objectlist
#        print "NGS coordinates = ", self.ngspos
#        print "LGS coordinates = ", self.lgspos
#        print "D pixels = ", self.pixdiam
#        print "CN2 profile = ", self.cn2
#        print "H profile = ", self.h_profile
#        print "Pupil D = ", self.pupil_diameter
#        print "dr0 = ", self.dr0
#        print "Zernike modes corrected= ", self.nz1
        otf_array = self.otf(objectlist, lambdaim, out=otfs, parallel=parallel, **kwargs)
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
                        hdu.header.update(key="pscale", value=self._scale(lambdaim), comment="arcsec per pixel")
                    hdu.writeto(out, clobber=True)
                else:
                    raise IOError("File must be of FITS format")
            except IOError:
                print "An error ocurred while writing to file "+str(out)
        return psf_array

    def _scale(self,lambdaim):
        """Compute plate scale in arc sec per pixel, lamda in nm"""
        theta = lambdaim * 1.e-9 / (2. * self.pupil_diameter)
        return 206264.8 * theta
        
