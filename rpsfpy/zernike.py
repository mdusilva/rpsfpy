import numpy as np
from scipy.misc import factorial as fact

def noll2zern(j):
    """Convert Noll index to Zernike radial and azimuthal indices"""
    
    if j <= 0:
        raise ValueError("Noll indices start at 1")
        
    n = np.trunc(np.sqrt(8*j - 7) - 1).astype(int) / 2
    if n % 2 == 0:
        m = 2 * ((j-(n*(n+1))/2) / 2)
    else:
        m = 1 + 2 * ((j-1-(n*(n+1))/2) / 2)
    return n, m

class Zernike(object):
    """Zernike polynomials"""
    
    def __init__(self, grid=None):
        self.grid = grid
        
    def rnm(self, rho, n, m):
        if n < 0: raise ValueError("n must be >= 0")
        elif m < 0: raise ValueError("m must be >= 0")
        elif n < m: raise ValueError("n must be >= m")
        elif (n-m)%2 == 0:
            R = 0.
            for s in np.arange(0, (n-m)/2+1):
                r = rho**(n-2*s)*((-1.)**s * fact(n-s)) / (fact(s)*fact((n+m)/2-s)*fact((n-m)/2-s))
                R = R + r
        else:
            R = 0.
        return R

    def zernike(self, rho, theta, j, orders=None):
        """
        normalized to pi
        orders should be a tuple
        theta in radians
        """
        if orders:
            n, m = orders
        else:
            n, m = noll2zern(j)
        R = self.rnm(rho, n, m)
        if m == 0:
            z = np.sqrt(n+1.) * R
#            return np.sqrt(n+1.) * R
            return z -z[0,0]
        elif j%2 == 0:
            z = np.sqrt(2. * (n+1.)) * R * np.cos(m*theta)
#            return np.sqrt(2. * (n+1.)) * R * np.cos(m*theta)
            return z - z[0,0]
        else:
            z = np.sqrt(2. * (n+1.)) * R * np.sin(m*theta)
#            return np.sqrt(2. * (n+1.)) * R * np.sin(m*theta)
            return z - z[0,0]
    
    def rotate(self, rho, theta, j, gamma=0., orders=None):
        """
        Apply rotation to zernike polynomial
        gamma and theta in radians
        """
        g = np.pi*gamma / 180.
        if orders:
            n, m = orders
        else:
            n, m = noll2zern(j)
        if m == 0:
#            z1 = self.zernike(rho, theta, j, orders=(n,m))
            return self.zernike(rho, theta, j, orders=(n,m))
#            return z1 - z1[0,0]
        if np.floor(n/2.)%2 == 1:
            if j%2 == 0:
                z1 = self.zernike(rho, theta, j, orders=(n,m))
                z2 = self.zernike(rho, theta, j-1, orders=(n,m))
                return z1 * np.cos(m*g) - z2 * np.sin(m*g)
            else:
                z1 = self.zernike(rho, theta, j, orders=(n,m))
                z2 = self.zernike(rho, theta, j+1, orders=(n,m))
                return z1 * np.cos(m*g) + z2 * np.sin(m*g)
        else:
            if j%2 == 0:
                z1 = self.zernike(rho, theta, j, orders=(n,m))
                z2 = self.zernike(rho, theta, j+1, orders=(n,m))
                return z1 * np.cos(m*g) - z2 * np.sin(m*g)
            else:
                z1 = self.zernike(rho, theta, j, orders=(n,m))
                z2 = self.zernike(rho, theta, j-1, orders=(n,m))
                return z1 * np.cos(m*g) + z2 * np.sin(m*g)
    
if __name__ == "__main__":
    
    zer = Zernike()
    
    for j in np.arange(1,38):
        print j, noll2zern(j)
