# -*- coding: utf-8 -*-
'''
Author   : Alexandre
Created  : 2020-10-21 08:33:26
Modified : 2020-10-21 15:44:46

Comments : testing odtcalc module
'''

# -- imports
from numpy import pi
from odtcalc.trap import Trap

# -- test
odt = Trap()
# PDH : aller
power = 4.8
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

# gradient coil
odt.magnetic_field_offset = (0, 0, 0)
odt.add_coil(plane='yz',
             radius=15e-2,
             axial_shift=30e-2,
             current=1,
             n_turns=100,
             label='gradient coil')

# bias coil
coil_1 = {'plane': 'yz',
          'radius': 10e-2,
          'axial_shift': 20e-2,
          'n_turns': 200,
          'current': 2}
coil_2 = {k: v for k, v in coil_1.items()}
coil_2['axial_shift'] = -coil_1['axial_shift']
#odt.add_coil_set([coil_1, coil_2], label='comp ODT')
odt.plot_potential(spatial_range=(1.5e-3, 500e-6, 500e-6))
odt.analyze_freq(plot_result=True, only_print_mean=True)
# odt.analyze_depth()
# res = odt.compute_theoretical_properties()
# print(res)