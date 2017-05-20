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

Inputs
------

Output
------

Example
-------

[1] Villecroze, R., "Modelisation d'un systeme d'optique adaptative a
grand champ pour la reconstruction de la reponse
impulsionnelle multi-spectrale des futurs
spectro-imageurs 3D du VLT et de l'ELT",  Lyon: Universite Claude Bernard Lyon 1, 2014.
"""
import os

import numpy as np
from scipy.special import jv
from scipy.integrate import trapz, quad, romberg, IntegrationWarning
from scipy import signal
import itertools
import zernike
try:
    import pyfits
except ImportError:
    import astropy.io.fits as pyfits
import json
import chassat

import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count

import copy_reg
from types import *
import warnings
import logging
import logging.handlers

warnings.filterwarnings("ignore", message="The integral is probably divergent, or slowly convergent.")
#create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

 # create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to debug
handler = logging.handlers.RotatingFileHandler(os.path.join(os.path.dirname(__file__), "debug.log"),"w", maxBytes=1024*1024, backupCount=1, delay="true")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

#zernike coefficients correlation mask
maskfile = os.path.join(os.path.dirname(__file__), r'masknn.fits')
z_handle = pyfits.open(maskfile)
zcoef_mask = z_handle[0].data

def _compmask(nz1, nz2):
    """Compute Zernike mask."""
    first = 2
    masku = np.zeros((nz1,nz2))
    for i in range(nz1):
        for j in range(i, nz2):
            masku[i,j] = (zernike.noll2zern(i+first)[1] % 2) == (zernike.noll2zern(j+first)[1] % 2)
    return masku.astype(int)


def _kvalues(n1=None, n2=None, m1=None, m2=None, z1=None, z2=None):
    """Compute K1 and K2 (cf. Chassat)"""
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
    """Inner integral to compute the correlations (cf. Chassat)"""
    #compute bessel functions
    j1 = jv(n1+1,x)
    j2 = jv(n2+1,w*x)
    j3 = jv(m1+m2,zeta*x)
    j4 = jv(np.abs(m1-m2),zeta*x)
    return x**(-14./3.) * j1 * j2 * (1. + (Lam / x)**2.)**(-11./6.) * (K1/w * j3 + K2/w * j4)

def _modified_chassat(t, zeta=None, Lam=None, w=None, K1=None, K2=None, n1=None, n2=None, m1=None, m2=None):
    """Chane of variables to deal with improper integral."""
    return _chassat_integral(np.tan(t), zeta, Lam, w, K1, K2, n1, n2, m1, m2) / np.cos(t)**2.

def fftcorr(X,Y,s=None):
    """Compute cross-correlation or autocorrelation using FFT.
    
    Parameters
    ----------
    X: numpy array
    Y: numpy array

    s: sequence of ints, optional
       shape of the output array, this passed to numpy's fft.fft2 function

    Returns
    -------

    out: numpy array
         Correlation matrix of X and Y
    """
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
    phi = np.arctan2(y, x+(rho<=0.))
    mask = np.where(np.logical_and(rho <= 1., rho >= oc), 1., 0.)
    if leq:
        mask = np.where(np.logical_and(rho <= 1., rho >= oc),1.,0.)
    else:
        mask = np.where(np.logical_and(rho < 1., rho >= oc),1.,0.)
    return rho, phi, mask

def _angletiti(x, y):
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
    def __init__(self, aoname, atmosphere, pixdiam, integrator, debug):
        self.aoname = aoname
        self.lgspos, self.hlgs, self.n_zernike, self.pupil_diameter = ao_systems.get(self.aoname)
        self.cn2 = np.array(atmosphere.cn2_profile)
        self.h_profile = np.array(atmosphere.h_profile)
        self.n_layers = len(self.cn2)
        self.outer_scale = atmosphere.outer_scale
        self.Zernike = zernike.Zernike()
        self.pixdiam = int(pixdiam)  #must be integer
        self.integrator = integrator
        self.rho, self.phi, self.mask = polar2(self.pixdiam/2., fourpixels=True, length=self.pixdiam)
        self._debug = debug

        if self.n_zernike > 980:
            self.zcoef_mask = _compmask(self.n_zernike, self.n_zernike).T
        else:
            self.zcoef_mask = zcoef_mask

    def cart2polar(self, objpos, gspos):
        """
        Convert cartesian coordinates to polar for the object
        and the GSs absolute and relative positions
        """
        ngs = len(gspos)
        combi = np.fromiter(itertools.combinations(np.arange(ngs), 2), np.dtype(('i, i')))
        ncombi = len(combi)
        xob, yob = objpos
        xgs, ygs = np.array(gspos).T
        alphags = np.zeros((ngs,ngs))
        gammags = np.zeros(ngs)
        betags = np.zeros(ncombi)
        for b_idx in np.arange(ncombi):
            l_idx = combi[b_idx][1]
            r_idx = combi[b_idx][0]
            betags[b_idx] = _angletiti(xgs[l_idx]-xgs[r_idx], ygs[l_idx]-ygs[r_idx])
        for i in np.arange(ngs):
            gammags[i] = _angletiti(xgs[i]-xob, ygs[i]-yob)
            for j in np.arange(i,ngs):
                if i == j:
                    alphags[i,j] = np.sqrt((xgs[i]-xob)**2. + (ygs[i]-yob)**2.)
                else:
                    alphags[i,j] = np.sqrt((xgs[i]-xgs[j])**2. + (ygs[i]-ygs[j])**2.)
                    alphags[j,i] = alphags[i,j]
        return alphags, gammags, betags

    def correl_swsw(self, angle, nz1, nz2, h1, h2, method='quad'):
        """Correlation coeficients between two spherical waves."""
        alpha = angle * 4.85*1.e-6
        R = self.pupil_diameter / 2.
        R1 = (h1-self.h_profile) / h1 * R
        R2 = (h2-self.h_profile) / h2 * R
        zeta = alpha  * self.h_profile / R1
        w = R2 / R1
        Lam = 2. * np.pi * R1 / self.outer_scale
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
        final_integral = 3.895 * (-1.)**((n1+n2-m1-m2)/2.) * np.sqrt((n1+1.)*(n2+1.)) * final_integral
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


class StructureLGS(Structure):
    def __init__(self, aoname, atmosphere, pixdiam, integrator, parallel='auto', hngs=10.e20, debug=False):
        super(StructureLGS, self).__init__(aoname, atmosphere, pixdiam, integrator, debug)
        self.nlgs = len(self.lgspos)
        self.hngs = hngs
        self.sigmaslgs = np.ones(self.nlgs) / self.nlgs
        self.combilgs = np.fromiter(itertools.combinations(np.arange(self.nlgs), 2), np.dtype(('i, i')))
        self.ncombilgs = len(self.combilgs)
        self.dphilgs = None
        self.zernikes = np.zeros((self.n_zernike-4, self.pixdiam, self.pixdiam))
        self.parallel = parallel
        if self.parallel == 'auto':
            self.cpus = cpu_count()
        else:
            self.cpus = int(parallel)
            if self.cpus < 1:
                raise ValueError("Must chose 1 or more CPUs")

    def setlgspos(self, lgspos):
        """Set LGS coordinates"""
        self.lgspos = lgspos

    def setobjpos(self, objpos):
        """Set Object coordinates"""
        self.objectpos = objpos

    def save2file(self, filename="dphilgs.dat"):
        """Save Structure function in ASCII file"""
        if self.dphilgs is not None:
            np.savetxt(filename, self.dphilgs)
        else:
            raise RuntimeError('LGS structure function not available')

    def readfromfile(self, filename):
        self.dphilgs = np.loadtxt(filename)

    def _compute_lgscoef(self, modes, h1, h2, alpha):
        n_modes = len(modes)
        coefmatrix = np.zeros(n_modes)
        for idx in xrange(n_modes):
            i = modes[idx][0]
            j = modes[idx][1]
            if self.integrator == "python":
                coefmatrix[idx] = self.correl_swsw(alpha, i, j, h1, h2)
            elif self.integrator == "idl":
                coefmatrix[idx] = chassat.correl_osos_general([alpha], self.pupil_diameter, h1, h2, self.cn2, self.h_profile, dr0 =1., num_zern1 = i, num_zern2 = j, gd_echelle = 25., borne_min = 7.e-3,npas = 100)
            else:
                raise ValueError("Integrator must be a choice of 'idl' or 'python' ")
        return coefmatrix, modes

    def cov_zernike(self, alpha, h1, h2):
        """
        Compute covariance matrix of Zernike coefficients

        The Zernike coefficients are from the expansions in Zernike basis of 
        two sources sepparated by alpha at heights h1 and h2
        """
        lgscoefmatrix = np.zeros((self.n_zernike-4, self.n_zernike-4))
        lgsnewzernikes = np.zeros((self.n_zernike-4, self.pixdiam, self.pixdiam))
        modes = [(i, j) for i in xrange(4, self.n_zernike) for j in xrange(4, self.n_zernike) if self.zcoef_mask[j-2, i-2]]
        #modes = []
        #for i in np.arange(4,self.n_zernike):
        #    for j in np.arange(i,self.n_zernike):
        #        if self.zcoef_mask[j-2,i-2]:
        #            modes.append((i,j))
        modes = np.array(modes)
        if self.cpus >= 1:
            pool = Pool(processes=self.cpus)
            start = 0
            end = len(modes)
            step = (end-start)/self.cpus + 1
            results = []
            for c in xrange(self.cpus):
                start_i = start + c*step
                end_i = min(start+(c+1)*step, end)
                modes_split = modes[start_i:end_i]
                results.append(pool.apply_async(self._compute_lgscoef, args=(modes_split, h1, h2, alpha)))
            pool.close()
            pool.join()
            for c in xrange(self.cpus):
                values_received, modes_received = results[c].get()
                for idx in xrange(len(modes_received)):
                    i = modes_received[idx][0]-4
                    j = modes_received[idx][1]-4
                    lgscoefmatrix[i, j] = values_received[idx]
        for i in xrange(4,self.n_zernike):
            for j in xrange(i,self.n_zernike):
                if j != i:
                    lgscoefmatrix[j-4,i-4] = lgscoefmatrix[i-4,j-4]
        return lgscoefmatrix

    def compute_term(self, alpha, gamma, h1, h2):
        """
        Compute a term of the LGS structure function corresponding to the
        correlation between two sources.
        """
        lgscoefmatrix = self.cov_zernike(alpha, h1, h2)
        if self._debug:
            fname = "coefmatrix" + "_LGS_" + "alpha_" + str(alpha) + "_gamma_" + str(gamma) + "_h1_" + str(h1) + "_h2_" + str(h2) + ".txt"
            logger.debug("DEBUG flag set: saving LGS covariance matrix in file %s, for sources sepparated by %s, at heights %s and %s", fname, str(alpha), str(h1), str(h2))
            np.savetxt(fname, lgscoefmatrix)
        propervalues, propervectors = self.diagcoef(lgscoefmatrix)
        for i in xrange(4, self.n_zernike):
            self.zernikes[i-4] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, gamma)
        lgsnewzernikes = np.dot(propervectors, self.zernikes.reshape((self.n_zernike-4, self.pixdiam*self.pixdiam))).reshape((self.n_zernike-4, self.pixdiam, self.pixdiam))
        lgsvmatrices = self.compvii(lgsnewzernikes, self.mask)
        return np.tensordot(propervalues, lgsvmatrices, axes=1)

    def lgs_term1(self):
        """Compute first term of LGS structure function"""
        logger.debug("Computing first term of LGS structure function.")
        term1 = self.compute_term(0., 0., self.hngs, self.hngs)
        return term1

    def lgs_term2(self):
        """Compute Second term of LGS structure function"""
        logger.debug("Computing second term of LGS structure function.")
        term2 = np.zeros((2*self.pixdiam, 2*self.pixdiam))
        for i_lgs in xrange(self.nlgs):
            term2 = term2 + self.sigmaslgs[i_lgs] * self.compute_term(self.alphalgs[i_lgs,i_lgs],-self.gammalgs[i_lgs], self.hngs, self.hlgs)
        return -2. * term2

    def lgs_term3(self):
        """Compute Third term of LGS structure function"""
        logger.debug("Computing third term of LGS structure function.")
        term3 = self.compute_term(0., 0., self.hlgs, self.hlgs)
        return term3 * np.sum(self.sigmaslgs**2.)

    def lgs_term4(self):
        """Compute Fourth term of LGS structure function"""
        logger.debug("Computing fourth term of LGS structure function.")
        term4 = np.zeros((2*self.pixdiam, 2*self.pixdiam))
        for b_lgs in xrange(self.ncombilgs):
            l_idx = self.combilgs[b_lgs][1]
            r_idx = self.combilgs[b_lgs][0]
            term4 = term4 + 2. * self.sigmaslgs[l_idx] * self.sigmaslgs[r_idx] * self.compute_term(self.alphalgs[l_idx,r_idx],-self.betalgs[b_lgs], self.hlgs, self.hlgs)
        return term4

    def compute(self, objpos):
        """Compute LGS structure function"""
        logger.info("Computing LGS structure function using %s CPUs.", str(self.cpus))
        self.objectpos = objpos
        self.alphalgs, self.gammalgs, self.betalgs = self.cart2polar(self.objectpos, self.lgspos)
        key1 = ("lgs1", self.aoname, self.n_layers, self.n_zernike, self.pixdiam, self.integrator)
        data1 = self.cn2.tolist() + self.h_profile.tolist() + [0., 0.]
        data1.append(self.outer_scale)
        term1 = pclib.getfunction(id_key=key1, data_values=data1)
        if term1 is None:
            term1 = self.lgs_term1()
            pclib.add(id_key=key1, data_values=data1, structure_function=term1)
        key2 = ("lgs2", self.aoname, self.n_layers, self.n_zernike, self.pixdiam, self.integrator)
        data2 = self.cn2.tolist() + self.h_profile.tolist() + np.diagonal(self.alphalgs).tolist() + self.gammalgs.tolist()
        data2.append(self.outer_scale)
        term2 = pclib.getfunction(id_key=key2, data_values=data2)
        if term2 is None:
            term2 = self.lgs_term2()
            pclib.add(id_key=key2, data_values=data2, structure_function=term2)
        key3 = ("lgs3", self.aoname, self.n_layers, self.n_zernike, self.pixdiam, self.integrator)
        data3 = self.cn2.tolist() + self.h_profile.tolist() + [0., 0.] 
        data3.append(self.outer_scale)
        term3 = pclib.getfunction(id_key=key3, data_values=data3)
        if term3 is None:
            term3 = self.lgs_term3()
            pclib.add(id_key=key3, data_values=data3, structure_function=term3)
        key4 = ("lgs4", self.aoname, self.n_layers, self.n_zernike, self.pixdiam, self.integrator)
        data4 = self.cn2.tolist() + self.h_profile.tolist() + [self.alphalgs[i,j] for (i,j) in self.combilgs] + self.betalgs.tolist()
        data4.append(self.outer_scale)
        term4 = pclib.getfunction(id_key=key4, data_values=data4)
        if term4 is None:
            term4 = self.lgs_term4()
            pclib.add(id_key=key4, data_values=data4, structure_function=term4)
        self.dphilgs = term1 + term2 + term3 + term4
        if self._debug:
            fname = "struct" + "_LGS_" + "object_" + str(self.objectpos) + ".txt"
            logger.debug("DEBUG flag set: saving LGS structure function in file %s, for sources at %s", fname, str(self.objectpos))
            np.savetxt(fname, self.dphilgs)
        return self.dphilgs


class StructureNGS(Structure):
    def __init__(self, aoname, atmosphere, pixdiam, integrator, hngs=10.e20, debug=False):
        super(StructureNGS, self).__init__(aoname, atmosphere, pixdiam, integrator, debug)
        self.hngs = hngs
        self.dphings = None
        self.zernikes = np.zeros((2, self.pixdiam, self.pixdiam))

    def setngspos(self, ngspos):
        """Set NGS coordinates"""
        self.ngspos = ngspos

    def setobjpos(self, objpos):
        """Set Object coordinates"""
        self.objectpos = objpos

    def save2file(self, filename="dphings.dat"):
        """Save Structure function in ASCII file"""
        if self.dphings is not None:
            np.savetxt(filename, self.dphings)
        else:
            raise RuntimeError('NGS structure function not available')

    def readfromfile(self, filename):
        self.dphings = np.loadtxt(filename)

    def _compute_ngscoef(self, modes, alpha):
        n_modes = len(modes)
        coefmatrix = np.zeros(n_modes)
        for idx in xrange(n_modes):
            i = modes[idx][0]
            j = modes[idx][1]
            if self.integrator == "python":
                coefmatrix[idx] = self.correl_swsw(alpha, i, j, self.hngs, self.hngs)
            elif self.integrator == "idl":
                coefmatrix[idx] = chassat.correl_osos_general([alpha], self.pupil_diameter, self.hngs, self.hngs, self.cn2, self.h_profile, dr0 =1., num_zern1 = i, num_zern2 = j, gd_echelle = 25., borne_min = 7.e-3,npas = 100)
            else:
                raise ValueError("Integrator must be a choice of 'idl' or 'python' ")
            #coefmatrix[idx] = self.correl_swsw(alpha, i, j, self.hngs, self.hngs)
            #coefmatrix[idx] = chassat.correl_osos_general([alpha], self.pupil_diameter, self.hngs, self.hngs, self.cn2, self.h_profile, dr0 =1., num_zern1 = i, num_zern2 = j, gd_echelle = 25., borne_min = 7.e-3,npas = 100)
        return coefmatrix, modes

    def cov_zernike(self, alpha):
        """
        Compute covariance matrix of Zernike coefficients

        The Zernike coefficients are from the expansions in Zernike basis of 
        two sources sepparated by alpha at heights h1 and h2
        """
        modes = [(i, j) for i in xrange(2, 4) for j in xrange(2, 4)]
        modes = np.array(modes)
        ngscoefmatrix = np.zeros((2, 2))
        ngsnewzernikes = np.zeros((2, self.pixdiam, self.pixdiam))
        values_received, modes_received = self._compute_ngscoef(modes, alpha)
        for idx in xrange(len(modes_received)):
            i = modes_received[idx][0] - 2
            j = modes_received[idx][1] - 2
            ngscoefmatrix[i, j] = values_received[idx]
            if j != i:
                ngscoefmatrix[j,i] = ngscoefmatrix[i,j]
        return ngscoefmatrix

    def compute_term(self, alpha, gamma):
        """
        Compute a term of the NGS structure function corresponding to the
        correlation between two sources.
        """
        ngscoefmatrix = self.cov_zernike(alpha)
        if self._debug:
            fname = "coefmatrix" + "NGS" + "alpha_" + str(alpha) + "_gamma_" + str(gamma) + "_h1_" + str(self.hngs) + "_h2_" + str(self.hngs) + ".txt"
            logger.debug("DEBUG flag set: saving NGS covariance matrix in file %s, for sources sepparated by %s, at heights %s and %s", fname, str(alpha), str(h1), str(h2))
            np.savetxt(fname, ngscoefmatrix)
        propervalues, propervectors = self.diagcoef(ngscoefmatrix)
        for i in xrange(2, 4):
            self.zernikes[i-2] = self.Zernike.rotate(self.mask*self.rho, self.phi, i, gamma)
        ngsnewzernikes = np.dot(propervectors, self.zernikes.reshape((2, self.pixdiam*self.pixdiam))).reshape((2, self.pixdiam, self.pixdiam))
        ngsvmatrices = self.compvii(ngsnewzernikes, self.mask)
        return np.tensordot(propervalues, ngsvmatrices, axes=1)

    def ngs_term1(self):
        """Compute first term of NGS structure function"""
        logger.debug("Computing first term of NGS structure function.")
        term1 = self.compute_term(0., 0.)
        return term1

    def ngs_term2(self):
        """Compute Second term of NGS structure function"""
        logger.debug("Computing second term of NGS structure function.")
        term2 = np.zeros((2*self.pixdiam, 2*self.pixdiam))
        for i_ngs in xrange(self.nngs):
            term2 = term2 + self.sigmasngs[i_ngs] * self.compute_term(self.alphangs[i_ngs,i_ngs], -self.gammangs[i_ngs])
        return -2. * term2

    def ngs_term3(self):
        """Compute Third term of NGS structure function"""
        logger.debug("Computing third term of NGS structure function.")
        term3 = self.compute_term(0., 0.)
        return term3 * np.sum(self.sigmasngs**2.)

    def ngs_term4(self):
        """Compute Fourth term of NGS structure function"""
        logger.debug("Computing fourth term of NGS structure function.")
        term4 = np.zeros((2*self.pixdiam, 2*self.pixdiam))
        for b_ngs in xrange(self.ncombings):
            l_idx = self.combings[b_ngs][1]
            r_idx = self.combings[b_ngs][0]
            term4 = term4 + 2. * self.sigmasngs[l_idx] * self.sigmasngs[r_idx] * self.compute_term(self.alphangs[l_idx,r_idx], -self.betangs[b_ngs])
        return term4

    def compute(self, objpos, ngspos):
        """Natural guide star anisoplanatism structure function"""
#        hngs = 10.e20 #try later with infinite
        logger.info("Computing NGS structure function.")
        self.objectpos = objpos
        self.ngspos = ngspos
        logger.debug("Ngs positions: %s", str(self.ngspos))
        self.nngs = len(self.ngspos)
        self.sigmasngs = np.ones(self.nngs) / self.nngs
        self.combings = np.fromiter(itertools.combinations(np.arange(self.nngs), 2), np.dtype(('i, i')))
        self.ncombings = len(self.combings)
        self.alphangs, self.gammangs, self.betangs = self.cart2polar(self.objectpos, self.ngspos)
        term1 = self.ngs_term1()
        term2 = self.ngs_term2()
        term3 = self.ngs_term3()
        term4 = self.ngs_term4()
        self.dphings = term1 + term2 + term3 + term4
        if self._debug:
            fname = "struct" + "_NGS_" + "object_" + str(self.objectpos) + "_ngs_" + str(self.ngspos) + ".txt"
            logger.debug("DEBUG flag set: saving NGS structure function in file %s, for sources at %s and NGS at %s", fname, str(self.objectpos), str(self.ngspos))
            np.savetxt(fname, self.dphings)
        return self.dphings


class Atmosphere(object):
    """
    Class representing an atmosphere

    Serves as a container of atmospheric variables
    """
    def __init__(self, rzero=None, cn2_profile=None, h_profile=None, outer_scale=None):
        """
        Initial values for the attributes.
        
        Parameters
        ----------
        rzero:
            turbulence coherence length.
        cn2_profile:
            Cn2 profile of turbulence.
        h_profile:
            height of atmospheric layers of Cn2 profile.
        outer_scale:
            turbulence outer scale.
        """
        self.rzero = rzero
        if len(cn2_profile) == len(h_profile):
            self.cn2_profile = cn2_profile
            self.h_profile = h_profile
        else:
            raise ValueError("Cn2 profile and height profile must be of same size")
        self.outer_scale = outer_scale

    def readfromfile(self, infile=None):
        """Not implemented"""
        return None


class AOsystem(object):
    """
    Class representing the AO system

    Handler of AO systems parameters.
    """
    def __init__(self, aofile="aofile.json"):
        self.aofile = os.path.join(os.path.dirname(__file__), aofile)
        self.systems = {}
        try:
            with open(self.aofile, "r") as f:
                self.systems.update(json.load(f))
        except ValueError:
            with open(self.aofile, "w+") as f:
                pass

    def view(self):
        """Print known AO systems"""
        for key in self.systems:
            print key
            print "------------"
            print "LGSs coordinates = ", self.systems[key][0]
            print "LGSs height = ", self.systems[key][1]
            print "Corrected Zernike modes = ", self.systems[key][2]
            print "Telescope diameter = ", self.systems[key][3]
            print "------------"

    def get(self, key):
        """Get parameters for a given AO system"""
        return self.systems[key]

    def add(self, name=None, lgspos=None, lgsheight=None, zmodes=None, diameter=None):
        """
        Initial values for the attributes.
        
        Parameters
        ----------
        name:
            string indicating name of AO system.
        lgspos:
            positions on the LGSs in arcsec.
        lgsheight:
            height of LGSs in m.
        zmodes:
            number of Zernike modes corrected by AO system.
        diameter:
            diameter of pupil in m.
        """
        if type(name) == str:
            self.systems[name] = lgspos, lgsheight, zmodes, diameter
            with open(self.aofile, 'w') as f:
                json.dump(self.systems, f)
        else:
            raise TypeError("Name of AO system must be a string.")
    
    def remove(self):
        """Not implemented"""
        return None


class Pclib(object):
    """
    Library of pre-computed structure functions
    """
    def __init__(self, libfile="pclib.json", libdir="pcdata"):
        self.libdir = os.path.join(os.path.dirname(__file__), libdir)
        self.libfile = os.path.join(self.libdir , libfile)
        self.sfuncs = {}
        try:
            with open(self.libfile,"r") as f:
                logger.debug("Loading Pre-computed libary to dictionary.")
                self.sfuncs.update(json.load(f))
        except (ValueError, IOError):
            with open(self.libfile,"w+") as f:
                logger.debug("Pre-computed library file does not exist, creating new...")
                pass
    
    def getfunction(self, id_key=None, data_values=None):
        """
        Get a structure function from the library.
        Returns None if the function is not in the library.
        """
        logger.debug("Retrieving structure function...")
        new_key = '_'.join((id_key[0], id_key[1], str(id_key[2]), str(id_key[3]), str(id_key[4]), str(id_key[5])))
        data_array = data_values
        if new_key in self.sfuncs:
            if id_key[0] == "fit":
                fname = os.path.join(self.libdir, self.sfuncs[new_key][0][0])
                structure_function = np.load(fname)
                logger.debug("Returning structure function for key %s.", new_key)
                return structure_function
            else:
                for elements in self.sfuncs[new_key]:
                    if np.allclose(np.array(data_array), np.array(elements[1])):
                        fname = os.path.join(self.libdir, elements[0])
                        structure_function = np.load(fname)
                        logger.debug("Returning structure function for key %s.", new_key)
                        return structure_function
        logger.debug("No structure function found for key %s", new_key)                
        return None
            
    def isknown(self, id_key=None, data_array=None):
        """Check if the structure function is already in the library"""
        new_key = '_'.join((id_key[0], id_key[1], str(id_key[2]), str(id_key[3]), str(id_key[4]), str(id_key[5])))
        if new_key in self.sfuncs:
            if id_key[0] == "fit":
                return True
            else:
                for elements in self.sfuncs[new_key]:
                    if np.allclose(np.array(data_array), np.array(elements[1])):
                        return True
        return False

    def remove(self, id_key=None):
        """
        Remove keys containing the input string.
        """
        keys_to_remove = []
        for key in self.sfuncs:
            if id_key in key:
                for elements in self.sfuncs[key]:
                    fname = os.path.join(self.libdir, elements[0])
                    try:
                        logger.debug("Removing file %s from library.", fname)
                        os.remove(fname)
                    except OSError:
                        logger.debug("file %s not found in library.", fname)
                        pass
                keys_to_remove.append(key)
        for key in keys_to_remove:
            logger.debug("Removing key %s from libary.", key)
            del self.sfuncs[key]
        with open(self.libfile, 'w') as f:
            logger.debug("Updating Pre-computed structure functions library file.")
            json.dump(self.sfuncs, f)
            
    def add(self, id_key=None, data_values=None, structure_function=None):
        """
        Add a pre-computed structure function to the database

        The id_key must be a tuple containing the variables:
        structure function identifier
        ao system name
        number of atmospheric layers
        number of corrected Zernike modes
        diameter of pupil in pixels
        Zernike integrator identifier

        the data_values are other parameters necessary to compute the 
        LGS sructure functions (the fitting error structure function
        can be computed with the information contained in id_key):
        cn2 profile
        height profile
        distance between sources
        angle between sources
        outer scale value

        """
        #new_key = (id_key[0], id_key[1], str(id_key[2]), str(id_key[3]), str(id_key[4]))
        new_key = '_'.join((id_key[0], id_key[1], str(id_key[2]), str(id_key[3]), str(id_key[4]), str(id_key[5])))
        #file_name = ''.join(new_key)
        file_name = os.path.join(self.libdir, new_key)
        data_array = data_values
        if self.isknown(id_key=id_key, data_array=data_array):
            logger.debug("Tried to add Structure function already in library.")
            return None
        else:
            if new_key in self.sfuncs:
                logger.debug("Adding new structure function to existing key %s", new_key)
                n_elements = len(self.sfuncs[new_key])
                new_file_name = file_name+"_"+str(n_elements)
                np.save(new_file_name, structure_function)
                self.sfuncs[new_key].append([new_key+"_"+str(n_elements)+".npy", data_array])
            else:
                logger.debug("Adding new structure function to new key %s", new_key)
                np.save(file_name, structure_function)
                self.sfuncs[new_key] = [[new_key+".npy", data_array]]
            with open(self.libfile, 'w') as f:
                logger.debug("Updating Pre-computed structure functions library file.")
                json.dump(self.sfuncs, f)
            return None


class StructureFit(Structure):
    """
    Fitting error structure function
    """
    def __init__(self, aoname, atmosphere, pixdiam, debug=False):
        super(StructureFit, self).__init__(aoname, atmosphere, pixdiam, "standard", debug)
        self.dphifit = None

    def save2file(self, filename="dphifit.dat"):
        """Save Structure function in ASCII file"""
        if self.dphifit is not None:
            np.savetxt(filename, self.dphifit)
        else:
            raise RuntimeError('Fitting error structure function not available')

    def readfromfile(self, filename):
        self.dphifit = np.loadtxt(filename)

    def _computefit(self, fc_constant):
        """Compute Fitting error structure function"""
        rho, phi, mask = polar2(self.pixdiam, length=2*self.pixdiam, center=[self.pixdiam,self.pixdiam])
        radial, azi = zernike.noll2zern(self.n_zernike)
        tot_correletion_aiaj = np.zeros((2*self.pixdiam,2*self.pixdiam))
        Fc = fc_constant * (radial + 1.)
        for i in range(2*self.pixdiam):
            for j in range(2*self.pixdiam):
                lower_bound = 2. * np.pi * Fc * rho[i,j]
                result_quad = quad(lambda x: x**(-8./3.)*(1.-jv(0,x)), lower_bound, 150.)
                tot_correletion_aiaj[i,j] = result_quad[0] * rho[i,j]**(5./3.)
        return 0.023 * 2.**(11./3.) * np.pi**(8./3.) * tot_correletion_aiaj

    def compute(self, fc_constant=0.37):
        """Fitting error structure function."""
        logger.info("Computing Fitting error structure function.")
        key = ("fit", self.aoname, self.n_layers, self.n_zernike, self.pixdiam, "standard")
        data = self.cn2.tolist() + self.h_profile.tolist() + [0., 0]
        data.append(self.outer_scale)
        dfit = pclib.getfunction(id_key=key, data_values=data)
        if dfit is None:
            dfit = self._computefit(fc_constant)
            pclib.add(id_key=key, data_values=data, structure_function=dfit)
        self.dphifit = dfit
        if self._debug:
            fname = "struct" + "_FIT" + ".txt"
            logger.debug("DEBUG flag set: saving Fitting error structure function in file %s", fname)
            np.savetxt(fname, self.dphifit)
        return self.dphifit


class Reconstruct(object):

    def __init__(self, pixdiam, ao_system, ngspos, atmosphere, parallel='auto', integrator='idl', debug=False):
        self.ao_system = ao_system
        self.ngspos = ngspos
        self.pixdiam = int(pixdiam)
        self.parallel = parallel
        self.integrator = integrator
        self.atmosphere = atmosphere
        self.Dngs = StructureNGS(self.ao_system, self.atmosphere, self.pixdiam, self.integrator, debug=debug)
        self.Dlgs = StructureLGS(self.ao_system, self.atmosphere, self.pixdiam, self.integrator, parallel=self.parallel, debug=debug)
        self.Dfit = StructureFit(self.ao_system, self.atmosphere, self.pixdiam, debug=debug)
        self.diameter = self.Dlgs.pupil_diameter
        logger.debug("Computing Structure functions with lgs positions: %s", str(self.Dlgs.lgspos))
        logger.debug("Computing Structure functions with lgs height: %s", str(self.Dlgs.hlgs))
        logger.debug("Computing Structure functions with Zernike modes: %s", str(self.Dlgs.n_zernike))
        logger.debug("Computing Structure functions with pupil diameter: %s", str(self.Dlgs.pupil_diameter))
        logger.debug("Computing Structure functions with Cn2 profile: %s", str(self.Dlgs.cn2))
        logger.debug("Computing Structure functions with height profile: %s", str(self.Dlgs.h_profile))
        logger.debug("Computing Structure functions with outer scale: %s", str(self.Dlgs.outer_scale))

    def strut_functions(self, objpos, dlgs=None, dngs=None, dfit=None):
        """Compute structure functions"""
        if dlgs is not None:
            self.dphilgs = np.loadtxt(dlgs)
        else:
            self.dphilgs = self.Dlgs.compute(objpos)
        if dngs is not None:
            self.dphings = np.loadtxt(dngs)
        else:
            self.dphings = self.Dngs.compute(objpos, self.ngspos)
        if dfit is not None:
            self.dfitting = np.loadtxt(dfit)
        else:
            self.dfitting = self.Dfit.compute()
        return self.dphilgs, self.dphings, self.dfitting

    def otf(self, objectlist, wavelength, out=None):
        """Compute OTF from Structrure functions
        
        This method is called to obtain estimates of OTFs for a given set of positions
        in the FoV and a given wavelength. By default, the OTFs are sampled at the Nyquist
        rate.

        Note that the default size of the image, in pixels, is an instance attribute.

        Parameters
        ----------
        objectlist: list of lists, tuples or arrays (2 elements)
                positions in the FoV (in arcsec) in which to estimate the OTF.
        wavelength: float
                wavelength of in nm
        out: string, optional
                Name of output file. If set, the output OTFs will be written to 
                a FITS file as a "data cube", i.e. the OTFs will be stacked in an
                3D array.
    
        Returns
        -------

        otf_array: ndarray
                OTFs stacked in a 3D Numpy array.
        """
        r0_500 = self.atmosphere.rzero
        r0_wv = r0_500 * (wavelength / 500.)**(6./5.)
        dr0 = (self.diameter / r0_wv) ** (5./3.)
        otf_array = []
        for objectpos in objectlist:
            logger.info("Computing Structure functions for object with coordinates %s", str(objectpos))
            dphilgs, dphings, dfitting = self.strut_functions(objectpos)
            dphi_tot = dphings * dr0 + dphilgs * dr0 + dfitting * dr0
            ac = fftcorr(self.Dlgs.mask,self.Dlgs.mask,s=(2*self.pixdiam, 2*self.pixdiam))
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
                    logger.info("Saving OTFs in file %s", out)
                    hdu.writeto(out, clobber=True)
                else:
                    logger.error("File must be of FITS format")
                    raise IOError("File must be of FITS format")
            except IOError:
                logger.error("An error ocurred while writing to file "+str(out))
        return otf_array

    def otf_fromfiles(self, wavelength):
        """Not implemented"""

    def psf(self, objectlist, wavelength, scale=1, out=None):
        """Compute PSF from Structrure functions.

        This method is called to obtain estimates of PSFs for a given set of positions
        in the FoV and a given wavelength. By default, the PSFs are sampled at the Nyquist
        rate. The PSFs can be outputted to a FITS file.

        Note that the default size of the image, in pixels, is an instance attribute.

        Parameters
        ----------
        objectlist: list of lists, tuples or arrays (2 elements)
                positions in the FoV (in arcsec) in which to estimate the PSF.
        wavelength: float
                wavelength of the PSF in nm
        scale: float, optional
                factor of increase in the PSF sampling rate relative to the Nyquist
                limit. default is 1 (Nyquist rate).
        out: string, optional
                Name of output file. If set, the output PSFs will be written to 
                a FITS file as a "data cube", i.e. the PSFs will be stacked in an
                3D array.
    
        Returns
        -------

        psf_array: ndarray
                PSFs stacked in a 3D Numpy array.
        """
        logger.info("Computing PSFs for wavelength = %s nm", str(wavelength))
        if scale < 1: raise ValueError("scale must be >= 1")
        psf_array = []
        otf_array = self.otf(objectlist, wavelength)
        for otf in otf_array:
            psf = self._resample(otf, scale)
            psf = psf / np.sum(psf)
            psf_array.append(psf)
        psf_array = np.array(psf_array)
        n1 = self.pixdiam * 2
        n2 = len(psf)
        if out:
            try:
                filename, fileExtension = os.path.splitext(out)
                if fileExtension == ".fits" or fileExtension == ".FITS":
                    hdu = pyfits.PrimaryHDU(psf_array)
                    try:
                        hdu.header['scale'] = (self._scale(wavelength)*(float(n1)/float(n2)), "arcsec per pixel")
                    except KeyError:
                        hdu.header.update(key="pscale", value=self._scale(wavelength)*(float(n1)/float(n2)), comment="arcsec per pixel")
                    logger.info("Saving PSFs in file %s", out)
                    hdu.writeto(out, clobber=True)
                else:
                    logger.error("File must be of FITS format")
                    raise IOError("File must be of FITS format")
            except IOError:
                logger.error("An error ocurred while writing to file "+str(out))
        return psf_array

    def _scale(self,wavelength):
        """Compute plate scale in arc sec per pixel, lamda in nm"""
        theta = wavelength * 1.e-9 / (2. * self.diameter)
        return 206264.8 * theta

    def _resample(self, otf, zoom):
        """Resample the PSF"""
        n_otf = self.pixdiam * 2
        n_pad = (n_otf * zoom - n_otf) / 2.
        n_pad = int(n_pad)
        a = np.pad(otf, n_pad, 'constant')
        a = np.fft.ifft2(a)
        a = np.fft.fftshift(a)
        return np.abs(a)

#Load library of pre-computed structure functions
pclib = Pclib()

#Load library of AO systems
ao_systems = AOsystem()
