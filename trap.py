# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-13 17:28:20
Modified : 2020-10-14 15:09:50

Comments : implements the Trap class, used for the calculation of optical
           dipole traps potential
'''
from atom import Helium

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


    def add_laser(self):
        pass


    def reset_lasers(self):
        self.lasers = []


    def compute_potential(self, X, Y, Z):
        pass





def DT_gauss(w0, P0):
    """
    Returns some parameters for the dipole trap, with waist w0 and power P0
    """
    global alpha, lbda, m_He
    zR = np.pi * w0 ** 2 / lbda
    I0 = 2 * P0 / np.pi / w0 ** 2
    U0 = 1 / 2 / csts.epsilon_0 / csts.c * alpha * I0 # trap depth
    omega_rad = np.sqrt( 4 * U0 / m_He / w0 ** 2) # radial trap freq.
    omega_ax = np.sqrt( 2 * U0 / m_He / zR ** 2) # axial trap freq.
    return U0, omega_rad, omega_ax


