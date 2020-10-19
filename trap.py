# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-13 17:28:20
Modified : 2020-10-19 18:05:17

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
from coil import SingleCoil
from utils import unit_mult, unit_str, \
    polyval2D, polyfit2D, sortp, analyze_psort

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
        self.magnetic_field_offset = (0, 0, 0)  # Tesla

    # -- components handling (lasers, coils)
    def add_laser(self, **kwargs):
        new_laser = GaussianBeam(**kwargs)
        self.lasers.append(new_laser)

    def reset_lasers(self):
        self.lasers = []

    def add_coil(self, **kwargs):
        new_coil = SingleCoil(**kwargs)
        self.coils.append(new_coil)

    def reset_coils(self):
        self.coils = []

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

        # compute magenetic field potential
        # prepare
        Bx0, By0, Bz0 = self.magnetic_field_offset  # homogeneous offset
        Bxc, Byc, Bzc = self.magnetic_field_offset  # magnetic field at center
        Bx = np.zeros_like(X * Y * Z, dtype=float) + Bx0
        By = np.zeros_like(X * Y * Z, dtype=float) + By0
        Bz = np.zeros_like(X * Y * Z, dtype=float) + Bz0
        gJ = self.atom.lande_g_factor
        mJ = self.atom.zeeman_state
        mu_B = csts.value('Bohr magneton')
        # add contributions of all coils
        for coil in self.coils:
            bx, by, bz = coil.field(X, Y, Z, unit='T')
            bxc, byc, bzc = coil.field(0, 0, 0, unit='T')
            if yield_each_contribution:
                b = np.sqrt((bx + Bx0)**2 + (by + By0)**2 + (bz + Bz0)**2)
                bc = np.sqrt((bxc + Bx0)**2 + (byc + By0)**2 + (bzc + Bz0)**2)
                mag_potential = mult * gJ * mJ * mu_B * (b - bc)
                individual_potentials[coil.label] = mag_potential
            Bx += bx
            By += by
            Bz += bz
            Bxc += bxc
            Byc += byc
            Bzc += bzc
        # compute resulting potential
        B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
        Bc = np.sqrt(Bxc ** 2 + Byc ** 2 + Bzc ** 2)  # at center
        magnetic_potential = gJ * mJ * mu_B * (B - Bc)
        potential += mult * magnetic_potential

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

    def compute_theoretical_properties(self):
        '''
        Computes and prints the theoretical properties (depth and frequencies)
        of each individual laser / coils.
        '''
        # -- loop on lasers
        for beam in self.lasers:
            # - compute
            # beam parameters
            w0 = beam.waist_value  # waist (m)
            zR = np.pi * w0 ** 2 / beam.wavelength  # Rayleigh length (m)
            P0 = beam.power  # power (W)
            I0 = 2 * P0 / np.pi / w0 ** 2  # central intensity (W/m^2)
            # atomic parameters
            alpha = self.atom.get_alpha(beam.wavelength)  # polarizability
            m = self.atom.mass  # atomic mass (kg)
            # trap parameters
            U0 = 1 / 2 / csts.epsilon_0 / csts.c * alpha * I0  # trap depth (J)
            U0_K = U0 / csts.k  # trap depth (K)
            omega_rad = np.sqrt(4 * U0 / m / w0 ** 2)  # radial trap freq.
            omega_ax = np.sqrt(2 * U0 / m / zR ** 2)  # axial trap freq.
            f_rad = omega_rad / 2 / pi
            f_ax = omega_ax / 2 / pi
            # - print
            print('>> Laser %s' % beam.label)
            print('  + depth    = %s' % unit_str(U0_K, prec=2, unit='K'))
            print('  + freq rad = %s' % unit_str(f_rad, prec=2, unit='Hz'))
            print('  + freq ax  = %s' % unit_str(f_ax, prec=2, unit='Hz'))
            print('')

        # -- loop on coils
        # TODO
        '''
        # -- theorie
        B0 = csts.mu_0 * I * n / 2 / R
        Bc = B0 * (1 + (z0/R)**2) ** (-3/2)
        grad = -3 * Bc * z0 / R**2 / (1 + (z0/R)**2)
        '''

        pass

    def _istrapping(self, U0, U):
        '''
        Determines whether the 2D potential U0 is trapping at level U
        It uses scikit-images find_countours() to check whether there is a
        closed contour at the level U. Returns 1 if trapping, -1 if not
        '''
        # find contours
        contours = find_contours(U, U0)
        result = -1
        # look for a closed contour
        for c in contours:
            if np.all(c[0, :] == c[-1, :]):  # if contour is closed
                result = 1
                break
        return result

    def analyze_depth(self,
                      spatial_range=(1.5e-3, 1e-3, 1e-3),
                      Npoint=(1000, 1000, 1000),
                      min_position=(0, 0, 0),
                      unit='µK',
                      plot_result=True,
                      style2D={'cmap': 'Spectral'},
                      print_result=True,
                      figsize=(12, 4)):
        '''
        Analyzes the trap potential to find its depth. The analysis is done on
        2D cuts, and is run three times, i.e. for the XY, XZ and YZ planes.
        The depth is determined using the _istrapping() method, i.e., by
        looking for a closed contour in the potential for a given energy.

        Parameters
        ----------
        spatial_range : tuple, optional
            spatial ranges (x,y,z) for the analysis, in meters
        Npoint : tuple, optional
            number of points for the (x,y,z) grids
        min_position : tuple, optional
            position of the trap minimum. center of the analysis area.
        unit : str, optional
            units for the potential : 'J', 'µK', 'mK', 'K'
        plot_result : bool, optional
            plot the results
        style2D : dict, optional
            style for 2D plot
        print_result : bool, optional
            prints the output of the analysis in the terminal

        Returns
        -------
        results : dictionnary
            dictionnary containing trap parameters with explicit names
        '''
        # -- prepare grids
        # 1D
        x = {}
        for i, ax in enumerate(['x', 'y', 'z']):
            x[ax] = np.linspace(-spatial_range[i], spatial_range[i], Npoint[i])
            x[ax] += min_position[i]
        # 2D grid
        XX = {}
        for cut in ['xy', 'xz', 'yz']:
            XX[cut] = np.meshgrid(x[cut[0]], x[cut[1]])
        # potential
        UU = {}
        x0, y0, z0 = min_position
        UU['xy'] = self.potential(XX['xy'][0], XX['xy'][1], z0, unit=unit)
        UU['xz'] = self.potential(XX['xz'][0], y0, XX['xz'][1], unit=unit)
        UU['yz'] = self.potential(x0, XX['yz'][0], XX['yz'][1], unit=unit)
        # trap minimum
        Umin = self.potential(x0, y0, z0, unit=unit)

        # -- analyze
        results = {}
        for cut in ['xy', 'xz', 'yz']:
            U = UU[cut]
            b = U.max()
            a = Umin + (b - Umin) * 1e-2
            if self._istrapping(a, U) == -1:
                Ulost = Umin
            elif self._istrapping(b, U) == 1:
                Ulost = b
                print('WARNING : could not find the depth in the %s cut' % cut)
                print(' you should increase the spatial range !')
            else:
                Ulost = brentq(self._istrapping, a, b,
                               args=(U), rtol=0.001, xtol=0.001)
            results[cut] = Ulost - Umin

        #  -- display
        if print_result:
            print('>> Depth analysis results:')
            for k, v in results.items():
                print(' + %s cut  = %.2g %s' % (k.upper(), v, unit))
            print('')

        # -- plot
        if plot_result:
            fig, ax = plt.subplots(1, 3, figsize=figsize)
            for cax, cut in zip(ax, ['xy', 'xz', 'yz']):
                # get grids and potential
                X = XX[cut][0]
                Y = XX[cut][1]
                xmult, xstr = unit_mult(X.max(), 'm')
                ymult, ystr = unit_mult(Y.max(), 'm')
                U = UU[cut]
                # plot potential
                pcm = cax.pcolormesh(xmult * X, ymult * Y, U,
                                     vmin=U.min(), vmax=U.max(),
                                     **style2D)
                fig.colorbar(pcm, ax=cax)
                # plot escaping contour
                Ulost = results[cut] + Umin
                contours = find_contours(U, Ulost)
                for c in contours:
                    cax.plot(xmult * x[cut[0]][np.uint(c[:, 1])],
                             ymult * x[cut[1]][np.uint(c[:, 0])], 'k')
                # setup
                cax.set_title('potential (%s)' % unit)
                cax.set_xlabel('%s (%s)' % (cut[0].upper(), xstr))
                cax.set_ylabel('%s (%s)' % (cut[1].upper(), ystr))

            # show
            plt.tight_layout()
            plt.show()

        return results

    def analyze_freq(self,
                     spatial_range=(60e-6, 20e-6, 20e-6),
                     Npoint=(200, 200, 200),
                     center=(0, 0, 0),
                     unit='µK',
                     plot_result=True,
                     style2D={'cmap': 'Spectral'},
                     print_result=True,
                     only_print_mean=False,
                     figsize=(12, 4)):
        '''
        Analyzes the trap potential to find trap center, frequencies and
        eigenaxes. The analysis is done on 2D cuts, and is run three times,
        i.e. for the XY, XZ and YZ planes

        Parameters
        ----------
        spatial_range : tuple, optional
            spatial ranges (x,y,z) for the analysis, in meters
        Npoint : tuple, optional
            number of points for the (x,y,z) grids
        center : tuple, optional
            center of the area to analyze
        unit : str, optional
            units for the potential : 'J', 'µK', 'mK', 'K'
        plot_result : bool, optional
            plot the results
        style2D : dict, optional
            style for 2D plot
        print_result : bool, optional
            prints the output of the analysis in the terminal

        Returns
        -------
        results : dictionnary
            dictionnary containing trap parameters with explicit names
        '''
        # -- prepare grids
        # 1D
        x = {}
        for i, ax in enumerate(['x', 'y', 'z']):
            x[ax] = np.linspace(-spatial_range[i], spatial_range[i], Npoint[i])
            x[ax] += center[i]
        # 2D grid
        XX = {}
        for cut in ['xy', 'xz', 'yz']:
            XX[cut] = np.meshgrid(x[cut[0]], x[cut[1]])
        # potential
        UU = {}
        x0, y0, z0 = center
        UU['xy'] = self.potential(XX['xy'][0], XX['xy'][1], z0, unit=unit)
        UU['xz'] = self.potential(XX['xz'][0], y0, XX['xz'][1], unit=unit)
        UU['yz'] = self.potential(x0, XX['yz'][0], XX['yz'][1], unit=unit)

        # -- analyze
        results = {}
        for cut in ['xy', 'xz', 'yz']:
            # 2D polynomial fit
            p = polyfit2D(XX[cut][0], XX[cut][1], UU[cut],
                          n=4, print_full_res=False)
            p_sorted = sortp(p)
            results[cut] = analyze_psort(p_sorted, unit=unit, m=self.atom.mass)
            results[cut]['p'] = p
            results[cut]['ps'] = p_sorted

        # -- display results
        if print_result:
            # format
            title = '>> Results for %s cut'
            res = '  + %s'
            mean = {'freq_x': [], 'freq_y': [], 'freq_z': [],
                    'x0': [], 'y0': [], 'z0': [], 'U0': []}
            # display
            for cut in ['xy', 'xz', 'yz']:
                # get results
                r = results[cut]
                theta = r['theta']
                theta_deg = theta * 180 / pi
                freq_u = unit_str(r['freq_u'], 2, 'Hz')
                freq_v = unit_str(r['freq_v'], 2, 'Hz')
                U0 = '%.2g %s' % (r['U0'], unit)
                x0 = '%.2g µm' % (r['x0'] * 1e6)
                y0 = '%.2g µm' % (r['y0'] * 1e6)
                # add to mean
                mean['freq_%s' % cut[0]].append(r['freq_u'])
                mean['freq_%s' % cut[1]].append(r['freq_v'])
                mean['%s0' % cut[0]].append(r['x0'])
                mean['%s0' % cut[1]].append(r['y0'])
                mean['U0'].append(r['U0'])
                # print
                if not only_print_mean:
                    print(title % cut.upper())
                    print(res % 'angle       = %.2g rad (%.2g deg)' %
                          (theta, theta_deg))
                    print(res % 'freq_u (~%s) = %s' % (cut[0], freq_u))
                    print(res % 'freq_v (~%s) = %s' % (cut[1], freq_v))
                    print(res % 'U0          = %s' % U0)
                    print(res % 'center %s   = %s' % (cut[0], x0))
                    print(res % 'center %s   = %s' % (cut[1], y0))
                    print('')
            # mean
            print('>> MEAN')
            for k in ['freq_x', 'freq_y', 'freq_z']:
                fm = np.mean(mean[k])
                print(res % '%s  = %s' % (k, unit_str(fm, 2, 'Hz')))
            for k in ['x0', 'y0', 'z0']:
                xm = np.mean(mean[k])
                print(res % '%s  = %.2g µm' % (k, xm))
            print(res % 'U0   = %.2g %s' % (np.mean(mean['U0']), unit))
            print('')

        # -- plot
        if plot_result:
            for cut in ['xy', 'xz', 'yz']:
                # - get grids and potentials
                X = XX[cut][0]
                Y = XX[cut][1]
                xmult, xstr = unit_mult(X.max(), 'm')
                ymult, ystr = unit_mult(Y.max(), 'm')

                U = UU[cut]
                Ufit = polyval2D(X, Y, results[cut]['p'])
                err = U - Ufit
                errmax = np.max(np.abs(err))
                theta = results[cut]['theta']
                x0 = results[cut]['x0']
                y0 = results[cut]['y0']
                # - plot
                # figure
                fig, ax = plt.subplots(1, 3, figsize=figsize)
                # potential
                pcm = ax[0].pcolormesh(xmult * X, ymult * Y, U,
                                       vmin=U.min(), vmax=U.max(),
                                       **style2D)
                fig.colorbar(pcm, ax=ax[0])
                ax[0].set_title('potential (%s)' % unit)
                # fit
                pcm = ax[1].pcolormesh(xmult * X, ymult * Y, Ufit,
                                       vmin=U.min(), vmax=U.max(),
                                       **style2D)
                fig.colorbar(pcm, ax=ax[1])
                ax[1].set_title('fit (%s)' % unit)
                # error
                pcm = ax[2].pcolormesh(xmult * X, ymult * Y, err,
                                       vmin=-errmax, vmax=errmax,
                                       cmap='RdBu_r')
                fig.colorbar(pcm, ax=ax[2])
                ax[2].set_title('error (%s)' % unit)

                # decorations
                for cax in ax:
                    # angles
                    r = np.array([-1, 1])
                    cax.plot((r * np.cos(theta) + x0) * xmult,
                             (r * np.sin(theta) + y0) * ymult,
                             label='u')
                    cax.plot((r * np.cos(theta + pi / 2) + x0) * xmult,
                             (r * np.sin(theta + pi / 2) + y0) * ymult,
                             label='v')
                    cax.plot(xmult * x0, ymult * y0, 'ok')
                    cax.set_xlim(xmult * X.min(), xmult * X.max())
                    cax.set_ylim(ymult * Y.min(), ymult * Y.max())
                    # labels
                    cax.set_xlabel('%s (%s)' % (cut[0].upper(), xstr))
                    cax.set_ylabel('%s (%s)' % (cut[1].upper(), ystr))

                ax[1].legend()
                plt.tight_layout()
            plt.show()

        return results

    # -- plotting

    def plot_potential(self,
                       spatial_range=(1e3, 1e3, 1e3),
                       Npoint=(500, 500, 500),
                       center=(0, 0, 0),
                       unit='µK',
                       style2D={'cmap': 'Spectral'},
                       style1D={},
                       Ncontour=6,
                       figsize=(11, 7)):
        '''
        Plots the total potential

        Parameters
        ----------
        spatial_range : tuple, optional
            spatial ranges (x,y,z) for the analysis, in meters
        Npoint : tuple, optional
            number of points for the (x,y,z) grids
        center : tuple, optional
            center of the area to analyze
        unit : str, optional
            units for the potential : 'J', 'µK', 'mK', 'K'
        style2D : dict, optional
            style for 2D plots
        style1D : dict, optional
            style for 1D plots
        Ncontour : int / array, optional
            number of contours to plot. One can also directly provide an array
            with the contours.
        '''

        # -- prepare grids
        # 1D
        x = {}
        for i, ax in enumerate(['x', 'y', 'z']):
            x[ax] = np.linspace(-spatial_range[i], spatial_range[i], Npoint[i])
            x[ax] += center[i]
        # 2D grid
        XX = {}
        for cut in ['xy', 'xz', 'yz']:
            XX[cut] = np.meshgrid(x[cut[0]], x[cut[1]])
        # 2D potential
        UU = {}
        x0, y0, z0 = center
        UU['xy'] = self.potential(XX['xy'][0], XX['xy'][1], z0, unit=unit)
        UU['xz'] = self.potential(XX['xz'][0], y0, XX['xz'][1], unit=unit)
        UU['yz'] = self.potential(x0, XX['yz'][0], XX['yz'][1], unit=unit)
        # 1D potential
        u = {}
        u['x'], u['x_ind'] = self.potential(x['x'], y0, z0, unit=unit,
                                            yield_each_contribution=True)
        u['y'], u['y_ind'] = self.potential(x0, x['y'], z0, unit=unit,
                                            yield_each_contribution=True)
        u['z'], u['z_ind'] = self.potential(x0, y0, x['z'], unit=unit,
                                            yield_each_contribution=True)

        # -- prepare contour
        if isinstance(Ncontour, (int, float)):
            Umin = UU['xy'].min()
            Umax = UU['xy'].max()
            contours = np.linspace(0, Umax - Umin, Ncontour)
        else:
            contours = Ncontour
        print('Contours : ')
        print(contours)

        # - plot
        # init figure
        plt.figure(figsize=figsize)
        ax = {}
        Ncol = 3
        Nrow = 2
        ax['xy'] = plt.subplot2grid((Nrow, Ncol), (0, 0))
        ax['xz'] = plt.subplot2grid((Nrow, Ncol), (0, 1))
        ax['yz'] = plt.subplot2grid((Nrow, Ncol), (0, 2))
        ax['x'] = plt.subplot2grid((Nrow, Ncol), (1, 0))
        ax['y'] = plt.subplot2grid((Nrow, Ncol), (1, 1))
        ax['z'] = plt.subplot2grid((Nrow, Ncol), (1, 2))
        # plot 2D
        for cut in ['xy', 'xz', 'yz']:
            cax = ax[cut]
            # get data
            X = XX[cut][0]
            Y = XX[cut][1]
            xmult, xstr = unit_mult(X.max(), 'm')
            ymult, ystr = unit_mult(Y.max(), 'm')
            U = UU[cut]
            # plot potential
            cax.pcolormesh(xmult * X, ymult * Y, U, **style2D)
            # plot contours
            cax.contour(xmult * X, ymult * Y, U, contours + Umin,
                        colors='k', linestyles='dashed', linewidths=1)
            cax.set_xlabel('%s (%s)' % (cut[0].upper(), xstr))
            cax.set_ylabel('%s (%s)' % (cut[1].upper(), ystr))
        # plot 1D
        for axis in ['x', 'y', 'z']:
            cax = ax[axis]
            r = x[axis]
            rmult, rstr = unit_mult(r.max(), 'm')
            # total potential
            cax.plot(rmult * r, u[axis], label='total', **style1D)
            # individual contributions
            for name, u_ind in u['%s_ind' % axis].items():
                cax.plot(rmult * r, u_ind, label=name, dashes=[2, 2])
            cax.set_ylabel('potential (%s)' % unit)
            cax.set_xlabel('%s (%s)' % (axis.upper(), rstr))
            cax.set_xlim(rmult * r.min(), rmult * r.max())
            cax.grid()
        # legend on one axis
        ax['x'].legend()
        # tight layout & show
        plt.tight_layout()
        plt.show()
        pass


# -- TESTS
if __name__ == '__main__':
    odt = Trap()
    # PDH : aller
    power = 6
    odt.add_laser(waist_value=135e-6,
                  power=power,
                  theta=pi / 2,  # i.e. in horizontal plane
                  phi=9 * pi / 180,
                  label='PDH (aller)')

    # PDH : retour
    odt.add_laser(waist_value=135e-6,
                  power=power * 0.8,
                  theta=pi / 2,
                  phi=-9 * pi / 180,
                  label='PDH (retour)')

    # magnetic
    odt.magnetic_field_offset = (4e-4, 0, 0)
    odt.add_coil(plane='yz',
                 radius=10e-2,
                 axial_shift=30e-2,
                 current=5,
                 n_turns=100,
                 label='gradient coil')

    odt.plot_potential(spatial_range=(1.5e-3, 500e-6, 500e-6))
    # odt.analyze_freq()
    odt.analyze_depth()
    # odt.compute_theoretical_properties()
