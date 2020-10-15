# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-13 17:28:20
Modified : 2020-10-15 15:01:56

Comments : implements the Trap class, used for the calculation of optical
           dipole traps potential
'''
# == imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy import constants as csts
from skimage.measure import find_contours
from scipy.optimize import brentq

# local
from atom import Helium
from laser import GaussianBeam
from utils import unit_mult

# == define atom data dictionnary
class Trap():
    '''
    Defines pulse shapes (temporal)
    '''
    def __init__(self, atom=Helium()):
        '''
        Object initialization, sets parameters
        '''
        self.atom = atom
        self.lasers = []
        self.coils = []
        self.gravity = True

    # -- components handling (lasers, coils)
    def add_laser(self, **kwargs):
        new_laser = GaussianBeam(**kwargs)
        self.lasers.append(new_laser)

    def reset_lasers(self):
        self.lasers = []

    # -- potential calculation
    def potential(self, X, Y, Z, yield_each_contribution=False, unit='J'):
        # check inputs
        unit_factor = {'J': 1,
                       'K': 1 / csts.k,
                       'mK': 1e3 / csts.k,
                       'µK': 1e6 / csts.k,
                       }
        assert unit in unit_factor.keys()
        mult = unit_factor[unit]
        # some declarations
        potential = np.zeros_like(X * Y * Z, dtype=float)
        if yield_each_contribution:
            individual_potentials = {}
        # compute optical potential
        for beam in self.lasers:
            intensity = beam.intensity(X, Y, Z)
            alpha = self.atom.get_alpha(wavelength=beam.wavelength)
            new_potential = - 0.5 / csts.epsilon_0 / csts.c * alpha * intensity
            new_potential *= mult
            potential += new_potential
            if yield_each_contribution:
                individual_potentials[beam.label] = new_potential
        # add gravity
        if self.gravity:
            m = self.atom.mass
            gravity_potential = m * csts.g * Z
            potential += mult * gravity_potential
        # return
        if yield_each_contribution:
            return potential, individual_potentials
        else:
            return potential

    # -- analyze
    def find_depth(self,
                   spatial_range=(1e3, 1e3, 1e3),
                   Npoints=(1000, 1000, 1000),
                   center=(0, 0, 0),
                   unit='µK',
                   plot_result=True,
                   style2D={'cmap': 'Spectral'}):
        pass


    # -- plotting
    def plot_potential(self,
                       spatial_range=(1e3, 1e3, 1e3),
                       Npoints=(1000, 1000, 1000),
                       center=(0, 0, 0),
                       unit='µK',
                       style2D={'cmap': 'Spectral'},
                       style1D={},
                       Ncontour=6):
        # - compute
        # get params
        Nx, Ny, Nz = Npoints
        xrange, yrange, zrange = spatial_range
        x0, y0, z0 = center
        # 1D arrays
        x = np.linspace(-xrange, xrange, Nx) + x0
        y = np.linspace(-yrange, yrange, Ny) + y0
        z = np.linspace(-zrange, zrange, Nz) + z0
        # grids
        XYx, XYy = np.meshgrid(x, y)  # XY
        XZx, XZz = np.meshgrid(x, z)  # XZ
        # compute 2D
        XYu = self.potential(XYx, XYy, z0, unit=unit)
        XZu = self.potential(XZx,  y0, XZz, unit=unit)
        # compute 1D
        ux, ux_ind = self.potential(x, y0, z0, unit=unit,
                                    yield_each_contribution=True)
        uy, uy_ind = self.potential(x0, y, z0, unit=unit,
                                    yield_each_contribution=True)
        uz, uz_ind = self.potential(x0, y0, z, unit=unit,
                                    yield_each_contribution=True)
        # - prepare for plot
        # scales for x,y,z
        xmult, xstr = unit_mult(xrange, 'm')
        ymult, ystr = unit_mult(yrange, 'm')
        zmult, zstr = unit_mult(zrange, 'm')
        # contour plot lines
        if isinstance(Ncontour, (int, float)):
            Umin = XYu.min()
            Umax = XYu.max()
            contours = np.linspace(0, Umax - Umin, Ncontour)
        else:
            contours = Ncontour
        print('Contours : ')
        print(contours)

        # - plot
        # init figure
        plt.figure(figsize=(11, 7))
        ax = {}
        Ncol = 3
        Nrow = 6
        ax['xy'] = plt.subplot2grid((Nrow, Ncol), (0, 0), rowspan=3)
        ax['xz'] = plt.subplot2grid((Nrow, Ncol), (3, 0), rowspan=3)
        ax['x'] = plt.subplot2grid((Nrow, Ncol), (0, 1), rowspan=2, colspan=2)
        ax['y'] = plt.subplot2grid((Nrow, Ncol), (2, 1), rowspan=2, colspan=2)
        ax['z'] = plt.subplot2grid((Nrow, Ncol), (4, 1), rowspan=2, colspan=2)
        # plot xy
        ax['xy'].pcolormesh(xmult * XYx, ymult * XYy, XYu, **style2D)
        ax['xy'].contour(xmult * XYx, ymult * XYy, XYu, contours + Umin,
                         colors='k', linestyles='dashed', linewidths=1)
        ax['xy'].set_xlabel('X (%s)' % xstr)
        ax['xy'].set_ylabel('Y (%s)' % ystr)
        # plot xz
        ax['xz'].pcolormesh(xmult * XZx, zmult * XZz, XZu, **style2D)
        ax['xz'].contour(xmult * XZx, zmult * XZz, XZu, contours + Umin,
                         colors='k', linestyles='dashed', linewidths=1)
        ax['xz'].set_xlabel('X (%s)' % xstr)
        ax['xz'].set_ylabel('Z (%s)' % zstr)
        # plot 1D cuts
        for a, xx, u, c, m, l in zip(['x', 'y', 'z'],
                                     [x, y, z],
                                     [ux, uy, uz],
                                     [ux_ind, uy_ind, uz_ind],
                                     [xmult, ymult, zmult],
                                     [xstr, ystr, zstr]):
            cax = ax[a]
            cax.plot(m * xx, u, label='total', **style1D)  # total contribution
            for name, u_ind in c.items():
                cax.plot(m * xx, u_ind, label=name, dashes=[2, 2])  # indivb

            cax.set_ylabel('potential (%s)' % unit)
            cax.set_xlabel('%s (%s)' % (a.upper(), l))
            cax.set_xlim(m * xx.min(), m * xx.max())
            cax.grid()
        ax['x'].legend()

        plt.tight_layout()
        plt.show()


# -- stored for later
def DT_gauss(w0, P0):
    """
    Returns some parameters for the dipole trap, with waist w0 and power P0
    """
    global alpha, lbda, m_He
    zR = np.pi * w0 ** 2 / lbda
    I0 = 2 * P0 / np.pi / w0 ** 2
    U0 = 1 / 2 / csts.epsilon_0 / csts.c * alpha * I0  # trap depth
    omega_rad = np.sqrt(4 * U0 / m_He / w0 ** 2)  # radial trap freq.
    omega_ax = np.sqrt(2 * U0 / m_He / zR ** 2)  # axial trap freq.
    return U0, omega_rad, omega_ax


# -- TESTS
if __name__ == '__main__':
    odt = Trap()
    # PDH : aller
    odt.add_laser(waist_value=135e-6,
                  power=6,
                  theta=pi / 2,  # i.e. in horizontal plane
                  phi=9 * pi / 180,
                  label='PDH (aller)')

    # PDH : retour
    odt.add_laser(waist_value=135e-6,
                  power=6 * 0.8,
                  theta=pi / 2,
                  phi=-9 * pi / 180,
                  label='PDH (retour)')

    odt.plot_potential(spatial_range=(1.5e-3, 500e-6, 500e-6))

