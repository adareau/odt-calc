# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-13 17:28:20
Modified : 2020-10-14 16:33:06

Comments : implements the Trap class, used for the calculation of optical
           dipole traps potential
'''
# -- imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy import constants as csts

# local
from atom import Helium
from laser import GaussianBeam

# -- define atom data dictionnary
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

    # - components handling (lasers, coils)
    def add_laser(self, **kwargs):
        new_laser = GaussianBeam(**kwargs)
        self.lasers.append(new_laser)

    def reset_lasers(self):
        self.lasers = []

    # - potential calculation
    def optical_potential(self, X, Y, Z, yield_for_each_laser=False):
        # some declarations
        potential = np.zeros_like(X * Y * Z)
        if yield_for_each_laser:
            individual_potentials = {}
        # compute potential
        for beam in self.lasers:
            intensity = beam.intensity(X, Y, Z)
            alpha = self.atom.get_alpha(wavelength=beam.wavelength)
            new_potential = 1 / 2 / csts.epsilon_0 / csts.c * alpha * intensity
            potential += new_potential
            if yield_for_each_laser:
                individual_potentials[beam.label] = new_potential

        # return
        if yield_for_each_laser:
            return potential, individual_potentials
        else:
            return potential

    # - plotting
    def plot_potential(self,
                       plot_range=(1e3, 1e3, 1e3),
                       Npoints=(1000, 1000, 1000),
                       center=(0, 0, 0),
                       laser=True,
                       coils=True):
        # get params
        Nx, Ny, Nz = Npoints
        xrange, yrange, zrange = plot_range
        x0, y0, z0 = center
        # 1D arrays
        x = np.linspace(-xrange, xrange, Nx)
        y = np.linspace(-yrange, yrange, Ny)
        z = np.linspace(-zrange, zrange, Nz)
        # grids
        XYx, XYy = np.meshgrid(x, y)  # XY
        XZx, XZz = np.meshgrid(x, z)  # XZ
        # compute

        # plot
        plt.figure(figsize=(13, 6))
        ax = {}
        Ncol = 3
        Nrow = 6
        ax['xy'] = plt.subplot2grid((Nrow, Ncol), (0, 0), rowspan=3)
        ax['xz'] = plt.subplot2grid((Nrow, Ncol), (3, 0), rowspan=3)
        ax['x'] = plt.subplot2grid((Nrow, Ncol), (0, 1), rowspan=2, colspan=2)
        ax['y'] = plt.subplot2grid((Nrow, Ncol), (2, 1), rowspan=2, colspan=2)
        ax['z'] = plt.subplot2grid((Nrow, Ncol), (4, 1), rowspan=2, colspan=2)
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

    odt.plot_potential()