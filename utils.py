# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-10-13 17:03:22
Modified : 2020-10-14 15:25:59

Comments : some utility functions for ODT potential calculation
'''
# -- imports
import numpy as np


# -- functions
def unit_str(x, prec=2, unit=''):
    if x == 0:
        return '0 %s' % unit
    # define units
    _disp_prefixes = {-18: 'a', -15: 'f', -12: 'p', -9: 'n', -6: 'µ', -3: 'm',
                      0: '', 3: 'k', 6: 'M', 9: 'G', 12: 'T'}
    # get range
    n = np.log(x) / np.log(10)
    k = n // 3
    if np.abs(n - 3 * (k + 1)) < 1e-10:
        k += 1
    power = np.clip(3 * k, -18, 12)
    y = x / 10**(power)
    # prepare string
    if y >= 100 and prec < 4:
        fmt = '%d %s%s'
    else:
        fmt = '%.' + str(prec) + 'g %s%s'
    out = fmt % (y, _disp_prefixes[int(power)], unit)
    return out
