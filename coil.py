# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-19 09:06:24
Modified : 2020-10-19 13:52:28

Comments : Implements the coils object, to compute coils magnetic fields
'''

# -- imports
import numpy as np
import scipy.constants as csts
from numpy import pi
from scipy.special import ellipe, ellipk


# -- functions
def mag_field_coil_xy(x, y, z, radius=1., current=1.):
    '''
    computes the magnetic field generated by a single coil located in the (x,y)
    plane, centered on (x=0, y=0, z=0).

    source :
    [1] https://mathcurve.com/courbes2d.gb/magneticcirculaire/article%20pre.pdf

    Parameters
    ----------
    x : float / array
        cartesian x-coordinate
    y : float / array
        cartesian y-coordinate
    z : float / array
        cartesian z-coordinate
    radius : float, optional
        coil radius (m)
    current : float, optional
        coil current (A)

    '''
    # -- convert x, y, z to arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # -- goes to spherical coordinates
    # rho and phi
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    # theta, avoiding origin
    mask = (rho == 0)
    rho_corrected = np.where(mask, 1, rho)
    theta = np.arccos(np.where(mask, 1, z / rho_corrected))
    # -- compute magnetic field
    # get notations from [1]
    a = radius
    r = rho * np.sin(theta)
    k = np.sqrt(4 * a * r / ((a + r) ** 2 + z ** 2))
    # compute
    A = csts.mu_0 * current / 2 / pi  # prefactor
    E = ellipe(k ** 2)  # complete elliptic integral of second kind
    K = ellipk(k ** 2)  # complete elliptic integral of first kind
    # along z
    Bz = 1 / np.sqrt((a + r) ** 2 + z ** 2) \
        * (K + (a**2 - r**2 - z**2) / ((a - r)**2 + z**2) * E)
    # along r (avoiding r=0)
    z_over_r = np.divide(z, r, where=(r != 0))
    z_over_r = np.clip(z_over_r, -1e10, 1e10)
    Br = z_over_r / np.sqrt((a + r) ** 2 + z ** 2) \
        * (-K + (a**2 + r**2 + z**2) / ((a - r)**2 + z**2) * E)

    # -- convert to x, y
    Bx = Br * np.cos(phi)
    By = Br * np.sin(phi)

    # -- in Tesla
    Bx *= A
    By *= A
    Bz *= A

    return (Bx, By, Bz)


# -- coils objects
class SingleCoil():
    '''
    A single coil, in a given plane (XY, XY or XZ)
    '''
    def __init__(self, **kwargs):
        '''
        Object initialization, sets parameters
        '''
        # -- initialize default settings
        # physical parameters
        self.n_turns = 100  # number of turns
        self.current = 1  # coil current (A)
        # geometry
        # NB: we consider that the coil lies in a given canonical plane
        #     (XY, XZ or YZ). The coil is centered - for instance, for a coil
        #     located in the XY plane, its center is located at (x=0, y=0) -
        #     and can be shifted in the direction orthogonal to the plane
        #     - for a coil in the XY plane, the position of the coil along the
        #     Z axis can be set using the 'axial_shift' parameter
        self.radius = 10e-2  # coil radius (m)
        self.axial_shift = 30e-2  # axial shift (m)
        self.plane = 'YZ'  # coil plane
        # other
        self.label = ''
        # -- initialize object
        # update attributes based on kwargs
        self.__dict__.update(kwargs)

    def field(self, x, y, z):
        '''
        returns intensity at point (x,y,z)
        '''
        # -- shorthand
        pass


# -- TESTS
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def B_coil(z, R, n, I):
        B_0 = csts.mu_0 * I * n / 2 / R
        alpha = np.arctan(z/R)
        return B_0 * np.cos(alpha) ** 3

    # - XY
    if False:
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        (Bx, By, Bz) = mag_field_coil_xy(X, Y, 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].pcolormesh(X, Y, Bz)
        ax[1].streamplot(X, Y, Bx, By, color='w')
        ax[1].pcolormesh(X, Y, np.sqrt(Bx**2 + By**2), vmin=0, vmax=1e-6)
        plt.show()

    # - XZ
    if False:
        x = np.linspace(-2, 2, 500)
        z = np.linspace(-2, 2, 500)
        X, Z = np.meshgrid(x, z)
        (Bx, By, Bz) = mag_field_coil_xy(X, 0, Z)
        B = np.sqrt(Bx**2 + Bz**2 + By**2)
        print(np.max(B))
        for Bi in [Bx, By, Bz]:
            print(np.max(np.abs(Bi)))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.streamplot(X, Z, Bx, Bz, color='w')
        ax.pcolormesh(X, Z, np.sqrt(Bx**2 + Bz**2), vmin=0, vmax=1e-6)
        plt.show()

    # - axis
    if True:
        z = np.linspace(-10, 10, 500)
        x = 0
        y = 0
        a = 0.5
        (Bx, By, Bz) = mag_field_coil_xy(0, 0, z, radius=a)
        Bth = csts.mu_0 * a ** 2 / 2 / (a**2 + z**2) ** (3/2)

        plt.figure()
        plt.plot(z, Bz, label='z')
        plt.plot(z, Bx, label='x')
        plt.plot(z, By, label='y')
        plt.plot(z, Bth, dashes=[2, 2], color='k', label='th')

        plt.tight_layout()
        plt.legend()
        plt.show()