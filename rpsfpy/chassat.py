"""
Functions to compute covariance between the Zernike coefficients of two spherical waves (cf. Chassat 1992)
"""

import numpy as np
from scipy.special import jv
from scipy.integrate import trapz, quad, romberg, IntegrationWarning
import zernike

def integrale_s_num(ff, alpha, hi, Rayon_s, RatioR, radial1, radial2, asimut1, asimut2, S1, S2, L0):
    j1 = jv(radial1+1,ff)
    j2 = jv(radial2+1,RatioR*ff)
    j3 = jv(asimut1+asimut2,ff*alpha*hi/Rayon_s)
    j4 = jv(abs(asimut1-asimut2),ff*alpha*hi/Rayon_s)
    integrand = ff**(-14/3.)*j1*j2* ((S1/RatioR)*j3+S2*j4)*(1.+(Rayon_s/(ff*L0/(2*np.pi)))**2.)**(-11./6.) 
    return integrand 

def calc_1_couche_s_num(alpha, hi, Rayon_s, RatioR, radial1, radial2, asimut1, asimut2, S1, S2, L0, nint, borne_min, borne_max):
    #fy, dlog = np.linspace(borne_min, borne_max, nint, retstep=True, endpoint=False)
    fy = np.logspace(np.log(borne_min), np.log(borne_max), nint, endpoint=False,  base=np.e)
    dlog = np.log(fy)[1] - np.log(fy)[0]
    dfy = fy*dlog
    intfgrd=np.sum(integrale_s_num(fy, alpha, hi, Rayon_s, RatioR, radial1, radial2, asimut1, asimut2, S1, S2, L0)*dfy)
    return intfgrd
    
def correl_osos_general(angle, diam, hsource1, hsource2, cn2, profil_h, dr0 =1., num_zern1 = 2, num_zern2 = 2, gd_echelle = 1., borne_min = 1.e-6, borne_max = 1000., npas = 1000):

    num_poly1 = num_zern1
    num_poly2 = num_zern2
    L0 = gd_echelle
    rayon = diam/2.   
    Rayon_spherique1 = (hsource1-profil_h)/hsource1*rayon
    Rayon_spherique2 = (hsource2-profil_h)/hsource2*rayon
    nangle = len(angle)
    nint = npas

    radial1, asimut1 = zernike.noll2zern(num_poly1)
    radial2, asimut2 = zernike.noll2zern(num_poly2)

    if radial1 == 1:
        borne_max = 100 
    else:
        borne_max = 500
    if borne_max > 1000:
        borne_max =  1000.

    if asimut1 == 0: 
        if asimut2 == 0: S1 = 1
        if asimut2 != 0 and (num_poly2 % 2) == 0: S1 = (-1.)**(asimut2)*np.sqrt(2.)
        if asimut2 != 0 and (num_poly2 % 2) != 0: S1 = 0
  
    if asimut1 != 0 and (num_poly1 % 2) == 0: 
        if asimut2 == 0: S1 = np.sqrt(2.)
        if asimut2 != 0 and (num_poly2 % 2) == 0: S1 = (-1.)**(asimut2)
        if asimut2 != 0 and (num_poly2 % 2) != 0: S1 = 0

    if asimut1 != 0 and (num_poly1 % 2) != 0:
        if asimut2 == 0: S1 = 0
        if asimut2 != 0 and (num_poly2 % 2) == 0: S1 = 0
        if asimut2 != 0 and (num_poly2 % 2) != 0: S1 = (-1.)**(asimut2+1)

    if asimut1 == 0: S2 =  0. 
    if asimut1 != 0 and (num_poly1 % 2) == 0:
        if asimut2 == 0: S2 = 0.
        if asimut2 != 0 and (num_poly2 % 2) == 0:
            if (asimut2+asimut1) % 2 == 0:
                S2 = 1 
            else: 
                if (asimut1-asimut2) < 0: 
                    S2 = -1 
                else:
                    S2 = 1
        if asimut2 != 0 and (num_poly2 % 2) != 0: S2 = 0

    if asimut1 != 0 and (num_poly1 % 2) != 0:
        if asimut2 == 0: S2 = 0.
        if asimut2 != 0 and (num_poly2 % 2) == 0: S2 = 0.
        if asimut2 != 0 and (num_poly2 % 2) != 0:
            if (asimut2+asimut1) % 2 == 0:
                S2 = 1.
            else: 
                if (asimut1-asimut2) < 0:
                    S2 = -1. 
                else:
                    S2 = 1.
  
    alpha = 0.
    correl_final = np.zeros(nangle) 
    bmin =  borne_min  
    bmax = bmin*100. 

    n_profil_h =  len(profil_h)
    for iii in range(0, nangle):
        correl =  0.
        alpha =  angle[iii]
        alpha =  alpha*4.85*1.e-6 

        for num in range(0, n_profil_h):
            borne_max1 =  borne_max 
            correl1 =  0.
            test = 0
            bmin =  borne_min
            bmax = bmin*100.
            hi = profil_h[num]
            Rayon_s = Rayon_spherique1[num]
            RatioR  = Rayon_spherique2[num]/Rayon_spherique1[num]

            if radial1 == 1:
                test = 1
                if hi*alpha/Rayon_s >= 10.:
                    borne_max1 = 100
                if hi*alpha/Rayon_s >= 50.:
                    borne_max1 = 100
                if hi*alpha/Rayon_s >= 100.:
                    borne_max1 = 50
                if hi*alpha/Rayon_s >= 500.:
                    borne_max1 = 20
                if hi*alpha/Rayon_s >= 1000.:
                    borne_max1 = 10
                if hi*alpha/Rayon_s >= 5000.:
                    borne_max1 = 5
                if hi*alpha/Rayon_s >= 20000.:
                    borne_max1 = 1
                if hi*alpha/Rayon_s >= 100000.:
                    borne_max1 = 0.1
                if hi*alpha/Rayon_s >= 500000.:
                    test = 0
            else:
                if radial1 == 2 and hi*alpha/Rayon_s <= 10.:
                    test = 1
                if radial1 == 3 and hi*alpha/Rayon_s <= 10.:
                    test = 1
                    borne_max = 300
                if radial1 == 4 and hi*alpha/Rayon_s <= 5.:
                    test = 1
                    borne_max = 200
                if radial1 == 5 and hi*alpha/Rayon_s <= 5.:
                    test = 1
                    borne_max = 200
                if radial1 == 6 and hi*alpha/Rayon_s <= 2.:
                    test = 1
                    borne_max = 100
                if radial1 == 7 and hi*alpha/Rayon_s <= 2.:
                    test = 1
                    borne_max = 100.
                if radial1 == 8 and hi*alpha/Rayon_s <= 2.:
                    test = 1
                    borne_max = 100.
                if radial1 == 9 and hi*alpha/Rayon_s <= 2.:
                    test = 1
                    borne_max = 50.
                if radial1 > 9 and hi*alpha/Rayon_s <= 2.:
                    test = 1
                    borne_max = 50.
            if test == 1: 
                if alpha <= 1.e-5 and (asimut1+asimut2) > 1.e-4 and np.abs(asimut1-asimut2) > 1.e-4:
                    correl1 =  0.
                else:
                    while True:
                        correl1 = correl1+ calc_1_couche_s_num(alpha, hi, Rayon_s, RatioR, radial1, radial2, asimut1, asimut2, S1, S2, L0, nint, bmin, bmax)
                        bmin =  bmax
                        bmax =  bmax*100.
                        if borne_max1/bmax < 100.: break
                    correl1 = correl1+calc_1_couche_s_num(alpha, hi, Rayon_s, RatioR, radial1, radial2, asimut1, asimut2, S1, S2, L0, nint, bmin, borne_max1)
                    correl1 =  correl1*cn2[num]*Rayon_s**(5./3.)
            correl = correl+correl1
        correl = 3.895*(-1.)**((radial1+radial2-asimut1-asimut2)/2)*np.sqrt((radial1+1.)*(radial2+1.))*(dr0)**(5./3.)*correl/np.sum(cn2*Rayon_spherique1**(5./3.))
        correl_final[iii] =  correl 
 
    return correl_final

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
    """Inner integral to compute the correlations (Chassat, 1992)"""
    #compute bessel functions
    j1 = jv(n1+1,x)
    j2 = jv(n2+1,w*x)
    j3 = jv(m1+m2,zeta*x)
    j4 = jv(np.abs(m1-m2),zeta*x)
    return x**(-14./3.) * j1 * j2 * (1. + (Lam / x)**2.)**(-11./6.) * (K1/w * j3 + K2/w * j4)

def correl_swsw(angle, nz1, nz2, h1, h2, pupil_diameter, cn2, h_profile, large_scale, method='quad'):
    """Correlation coeficients between two spherical waves."""
    alpha = angle * 4.85*1.e-6
    R = pupil_diameter / 2.
    R1 = (h1-h_profile) / h1 * R
    R2 = (h2-h_profile) / h2 * R
    zeta = alpha  * h_profile / R1
    w = R2 / R1
    Lam = 2. * np.pi * R1 / large_scale
    n1, m1 = zernike.noll2zern(nz1)
    n2, m2 = zernike.noll2zern(nz2)
    k1, k2 = _kvalues(n1, n2, m1, m2, nz1, nz2)
    results = np.zeros(len(h_profile))
    for idx in np.arange(len(h_profile)):
        if method == 'quad':
            result_quad = quad(_chassat_integral, 0, np.inf, args=(zeta[idx], Lam[idx], w[idx],  k1, k2, n1, n2, m1, m2))
            results[idx] = result_quad[0] * cn2[idx] * R1[idx]**(5./3.)
        elif method == 'romberg':
            result_quad = romberg(_modified_chassat, 1.e-26,np.pi/2., args=(zeta[idx], Lam[idx], w[idx], k1, k2, n1, n2, m1, m2), vec_func = False)
            results[idx] = result_quad * cn2[idx] * R1[idx]**(5./3.)
    if len(results) < 2:
        final_integral = results / (cn2 * R1**(5./3.))
    else:
        final_integral = trapz(results, x=h_profile) / trapz(cn2 * R1**(5./3.), x=h_profile)
    final_integral = 3.895 * (-1.)**((n1+n2-m1-m2)/2.) * np.sqrt((n1+1.)*(n2+1.)) * final_integral
    return final_integral

if __name__ == "__main__":
    import time
    t0 = time.time()
    print correl_osos_general([20.], 8., 1000000000000000000000., 90000., np.array([0.7, 0.3]), np.array([0., 10000.]), dr0 =1., num_zern1 = 6, num_zern2 = 6, gd_echelle = 25., borne_min = 7.e-3,npas = 100)
    print time.time() - t0
    t0 = time.time()
    print correl_swsw(20., 6, 6, 1000000000000000000000.,  90000., 8. , np.array([0.7, 0.3]), np.array([0., 10000.]), 25., method='quad')
    print time.time() - t0