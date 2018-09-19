#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# ====================
# @file: param.py
# @author: zyt
# @time: 2018/05/17
# ====================
"""
calculate teff from the color indices, considering the galactic dust reddening and extinction
e(b-v) through the coordinate from the website: http://irsa.ipac.caltech.edu/applications/dust/.
extinction measurement refered to schlegel, finkbeiner & davis (1998); schlafly and finkbeiner (2011)(default).
empirical teff-color relations refered to alonso 1996,1999; ramirez 2005; casagrande 2010.
"""
import numpy as np
import pandas as pd
#======================================================================================================================
def cal_teff(jmag, hmag, ksmag, bmag, vmag, e_bv, feh=0.0, mode=0, star_type='giant', display=True):
    """
    :param jmag: 2mass magnitude system
    :param hmag: 2mass magnitude system
    :param ksmag: 2mass magnitude system
    :param bmag: johnson system
    :param vmag: johnson system
    :param e_bv:
    :param feh: default is 0.0
    :param mode: mode = 0 (default) for v-k,  1 for v-j and v-h,  2 for b-v
    :param star_type: giant or dwarf
    :return: teff
    2005Ramirez calibrations range from F0 to K5 (4000 K ~ 7000 K), metallicity range -3.5 ~ 0.4
    """
    teff = np.zeros(3)
    bmv = bmag - vmag
    bmv_0 = bmv - e_bv

    e_vmj = e_bv * 2.16  # ramirez2005ii
    vmj = vmag - jmag
    vmj_0 = vmj - e_vmj

    e_vmh = e_bv * 2.51  # ramirez2005ii
    vmh = vmag - hmag
    vmh_0 = vmh - e_vmh

    e_vmk = e_bv * 2.70  # ramirez2005ii
    vmk = vmag - ksmag
    vmk_0 = vmk - e_vmk  # here k represent ks from 2mass

    e_vmktcs = e_bv * 2.74  # 2005ramirezi

# vmktcs_1 = 0.05 + 0.993*vmk_0 # alonso1998
    a99 = [[0.5558, 0.2105, 1.981e-3, -9.965e-3, 1.325e-2, -2.726e-3],
            [0.3770, 0.3660, -3.170e-2, -3.074e-3, -2.765e-3, -2.973e-3],
            [0.5716, 0.5404, -6.126e-2, -4.862e-2, -1.777e-2, -7.969e-3],
            [0.6177, 0.4354, -4.025e-3, 5.204e-2, -0.1127, -1.385e-2]]
    a96 = [[0.555, 0.195, 0.013, -0.008, 0.009, -0.002],
            [0.566, 0.217, -0.003, -0.024, 0.037, -0.002],
            [0.541, 0.533, 0.007, -0.019, -0.047, -0.011]]
    c10 = [[0.5057, 0.2600, -0.0146, -0.0131, 0.0288, 0.0016],
            [0.5665, 0.4809, -0.0060, -0.0613, -0.0042, -0.0055]]
    r05g = [[0.4405, 0.3272, -0.0252, -0.0016, -0.0053, -0.0040],
            [0.2943, 0.5604, -0.0677, 0.0179, -0.0532, -0.0088],
            [0.4354, 0.3405, -0.0263, -0.0012, -0.0049, -0.0027],
            [0.5737, 0.4882, -0.0149, 0.0563, -0.1160, -0.0114]]
    r05d = [[0.4942, 0.2809, -0.0180, -0.0294, 0.0444, -0.0008],
            [0.4050, 0.4792, -0.0617, -0.0392, 0.0401, -0.0023],
            [0.4931, 0.3056, -0.0241, -0.0396, 0.0678, 0.0020],
            [0.5002, 0.6440, -0.0690, -0.0230, -0.0566, -0.0170]]
    cor_r05g_vmk = [[-72.6664, 36.5361, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [86.0358, -65.4928, 10.8901, 0.0, 0.0, 0.0, 0.0],
                    [-6.96153, 14.3298, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-943.925, 1497.64, -795.867, 138.965, 0.0, 0.0, 0.0]]
    cor_r05g_vmj = [[-122.595, 76.4847, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-10.3848, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [4.18695, 13.8937, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-67.7716, 28.9202, 0.0, 0.0, 0.0, 0.0, 0.0]]
    cor_r05g_vmh = [[-377.022, 334.733, -69.8093, 0.0, 0.0, 0.0, 0.0],
                    [71.7949, -55.5383, 9.61821, 0.0, 0.0, 0.0, 0.0],
                    [-27.4190, 20.7082, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-46.2946, 20.1061, 0.0, 0.0, 0.0, 0.0, 0.0]]
    cor_r05g_bmv = [[112.116, -372.622, 67.1254, 395.333, -203.471, 0.0, 0.0],
                    [-12.9762, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [606.032, -1248.79, 627.453, 0.0, 0.0, 0.0, 0.0],
                    [-9.26209, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    cor_r05d_vmk = [[-1425.36, 3218.36, -2566.54, 859.644, -102.554, 0.0, 0.0],
                    [2.35133, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1849.46, 4577.00, -4284.02, 1700.38, -268.589, 0.0, 0.0],
                    [215.721, -796.519, 714.423, -175.678, 0.0, 0.0, 0.0]]
    cor_r05d_vmj = [[422.406, -910.603, 621.335, -132.566, 0.0, 0.0, 0.0],
                    [-466.616, 658.349, -220.454, 0.0, 0.0, 0.0, 0.0],
                    [-862.072, 1236.84, -423.729, 0.0, 0.0, 0.0, 0.0],
                    [-1046.10, 1652.06, -597.340, 0.0, 0.0, 0.0, 0.0]]
    cor_r05d_vmh = [[-53.5574, 36.0990, 15.6878, -8.84468, 0.0, 0.0, 0.0],
                    [1.60629, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [506.559, -1277.52, 939.519, -208.621, 0.0, 0.0, 0.0],
                    [-471.588, 643.972, -199.639, 0.0, 0.0, 0.0, 0.0]]
    cor_r05d_bmv = [[-261.548, 684.977, -470.049, 79.8977, 0.0, 0.0, 0.0],
                    [-324.033, 1516.44, -2107.37, 852.150, 0.0, 0.0, 0.0],
                    [30.5985, -46.7882, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [139.965, -292.329, 0.0, 0.0, 0.0, 0.0, 0.0]]

    def formula_teff(a0, a1, a2, a3, a4, a5, col_index, feh):
        theta_eff = a0 + a1*col_index + a2*col_index**2 + a3*col_index*feh + a4*feh + a5*feh**2
        return 5040./theta_eff

    def cor_r05(p0, p1, p2, p3, p4, p5, p6, col_index):
        peff = p0 + p1*col_index + p2*col_index**2 + p3*col_index**3 + p4*col_index**4 + p5*col_index**5 \
               + p6*col_index**6
        return peff
    if feh in [np.nan, 99.0]:
        feh = 0.0
    if star_type == 'giant':
        ktcs = ksmag - 0.014 + 0.027 * (jmag - ksmag)  # 2005ramirezi
        vmktcs = vmag - ktcs
        vmktcs_0 = vmktcs - e_vmktcs
        if mode == 0:
            if (vmktcs_0 >= 0.20) & (vmktcs_0 <= 2.50):
                teff[0] = formula_teff(a99[0][0], a99[0][1], a99[0][2], a99[0][3], a99[0][4],
                                       a99[0][5], vmktcs_0, feh)
                if display:
                    print 'giant v-k (alonso1999) eq.8', '{:.0f}'.format(teff[0])
            else:
                teff[0] = formula_teff(a99[1][0], a99[1][1], a99[1][2], a99[1][3], a99[1][4],
                                      a99[1][5], vmktcs_0, feh)
                if display:
                    print 'giant v-k (alonso1999) eq.9', '{:.0f}'.format(teff[0])

            teff_a = formula_teff(r05g[0][0], r05g[0][1], r05g[0][2], r05g[0][3],
                                  r05g[0][4], r05g[0][5], vmk_0, feh)
            if (vmk_0 >= 1.244) & (vmk_0 <= 3.286) and (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_vmk[0][0], cor_r05g_vmk[0][1], cor_r05g_vmk[0][2], cor_r05g_vmk[0][3],
                                  cor_r05g_vmk[0][4], cor_r05g_vmk[0][5], cor_r05g_vmk[0][6], vmk_0)
                if display:
                    print 'giant v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.366) & (vmk_0 <= 4.474) and (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_vmk[1][0], cor_r05g_vmk[1][1], cor_r05g_vmk[1][2], cor_r05g_vmk[1][3],
                                  cor_r05g_vmk[1][4], cor_r05g_vmk[1][5], cor_r05g_vmk[1][6], vmk_0)
                if display:
                    print 'giant v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.334) & (vmk_0 <= 3.549) and (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_vmk[2][0], cor_r05g_vmk[2][1], cor_r05g_vmk[2][2], cor_r05g_vmk[2][3],
                                  cor_r05g_vmk[2][4], cor_r05g_vmk[2][5], cor_r05g_vmk[2][6], vmk_0)
                if display:
                    print 'giant v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.258) & (vmk_0 <= 2.768) and (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_vmk[3][0], cor_r05g_vmk[3][1], cor_r05g_vmk[3][2], cor_r05g_vmk[3][3],
                                  cor_r05g_vmk[3][4], cor_r05g_vmk[3][5], cor_r05g_vmk[3][6], vmk_0)
                if display:
                    print 'giant v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 v-k range!'
            return '{:07.2f}  {:07.2f}'.format(teff[0], teff[1])

        if mode == 1:
            teff_a = formula_teff(r05g[1][0], r05g[1][1], r05g[1][2], r05g[1][3],
                                  r05g[1][4], r05g[1][5], vmj_0, feh)
            if (vmj_0 >= 1.259) & (vmj_0 <= 2.400) & (feh >= -0.5) & (feh <= 0.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05g_vmj[0][0], cor_r05g_vmj[0][1], cor_r05g_vmj[0][2], cor_r05g_vmj[0][3],
                                  cor_r05g_vmj[0][4], cor_r05g_vmj[0][5], cor_r05g_vmj[0][6], vmj_0)
                if display:
                    print 'giant v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 1.030) & (vmj_0 <= 3.418) & (feh >= -1.5) & (feh <= -0.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05g_vmj[1][0], cor_r05g_vmj[1][1], cor_r05g_vmj[1][2], cor_r05g_vmj[1][3],
                                  cor_r05g_vmj[1][4], cor_r05g_vmj[1][5], cor_r05g_vmj[1][6], vmj_0)
                if display:
                    print 'giant v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 1.033) & (vmj_0 <= 2.679) & (feh >= -2.5) & (feh <= -1.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05g_vmj[2][0], cor_r05g_vmj[2][1], cor_r05g_vmj[2][2], cor_r05g_vmj[2][3],
                                  cor_r05g_vmj[2][4], cor_r05g_vmj[2][5], cor_r05g_vmj[2][6], vmj_0)
                if display:
                    print 'giant v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 0.977) & (vmj_0 <= 2.048) & (feh >= -4.0) & (feh <= -2.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05g_vmj[3][0], cor_r05g_vmj[3][1], cor_r05g_vmj[3][2], cor_r05g_vmj[3][3],
                                  cor_r05g_vmj[3][4], cor_r05g_vmj[3][5], cor_r05g_vmj[3][6], vmj_0)
                if display:
                    print 'giant v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            else:
                print 'out of ramirez2005 v-j range!'

            teff_b = formula_teff(r05g[2][0], r05g[2][1], r05g[2][2], r05g[2][3],
                                  r05g[2][4], r05g[2][5], vmh_0, feh)
            if (vmh_0 >= 1.194) & (vmh_0 <= 3.059) and (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05g_vmh[0][0], cor_r05g_vmh[0][1], cor_r05g_vmh[0][2], cor_r05g_vmh[0][3],
                                  cor_r05g_vmh[0][4], cor_r05g_vmh[0][5], cor_r05g_vmh[0][6], vmh_0)
                if display:
                    print 'giant v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.293) & (vmh_0 <= 4.263) & (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05g_vmh[1][0], cor_r05g_vmh[1][1], cor_r05g_vmh[1][2], cor_r05g_vmh[1][3],
                                  cor_r05g_vmh[1][4], cor_r05g_vmh[1][5], cor_r05g_vmh[1][6], vmh_0)
                if display:
                    print 'giant v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.273) & (vmh_0 <= 3.416) & (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05g_vmh[2][0], cor_r05g_vmh[2][1], cor_r05g_vmh[2][2], cor_r05g_vmh[2][3],
                                  cor_r05g_vmh[2][4], cor_r05g_vmh[2][5], cor_r05g_vmh[2][6], vmh_0)
                if display:
                    print 'giant v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.232) & (vmh_0 <= 2.625) & (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05g_vmh[3][0], cor_r05g_vmh[3][1], cor_r05g_vmh[3][2], cor_r05g_vmh[3][3],
                                  cor_r05g_vmh[3][4], cor_r05g_vmh[3][5], cor_r05g_vmh[3][6], vmh_0)
                if display:
                    print 'giant v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 v-h range!'
            return '{:07.2f}  {:07.2f}'.format(teff[0], teff[1])

        if mode == 2:
            if (bmv_0 >= 0.20) & (bmv_0 <= 0.80):
                teff[0] = formula_teff(a99[2][0], a99[2][1], a99[2][2], a99[2][3], a99[2][4],
                                       a99[2][5], bmv_0, feh)
                if display:
                    print 'giant b-v (alonso1999) eq.3', '{:.0f}'.format(teff[0])
            else:
                teff[0] = formula_teff(a99[3][0], a99[3][1], a99[3][2], a99[3][3], a99[3][4],
                                       a99[3][5], bmv_0, feh)
                if display:
                    print 'giant b-v (alonso1999) eq.4', '{:.0f}'.format(teff[0])

            teff_a = formula_teff(r05g[3][0], r05g[3][1], r05g[3][2], r05g[3][3],
                                  r05g[3][4], r05g[3][5], bmv_0, feh)
            if (bmv_0 >= 0.144) & (bmv_0 <= 1.668) and (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_bmv[0][0], cor_r05g_bmv[0][1], cor_r05g_bmv[0][2], cor_r05g_bmv[0][3],
                                  cor_r05g_bmv[0][4], cor_r05g_bmv[0][5], cor_r05g_bmv[0][6], bmv_0)
                if display:
                    print 'giant b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0.664) & (bmv_0 <= 1.558) & (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_bmv[0][0], cor_r05g_bmv[0][1], cor_r05g_bmv[0][2], cor_r05g_bmv[0][3],
                                  cor_r05g_bmv[0][4], cor_r05g_bmv[0][5], cor_r05g_bmv[0][6], bmv_0)
                if display:
                    print 'giant b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0605) & (bmv_0 <= 1.352) & (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_bmv[0][0], cor_r05g_bmv[0][1], cor_r05g_bmv[0][2], cor_r05g_bmv[0][3],
                                  cor_r05g_bmv[0][4], cor_r05g_bmv[0][5], cor_r05g_bmv[0][6], bmv_0)
                if display:
                    print 'giant b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0.680) & (bmv_0 <= 1.110) & (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05g_bmv[3][0], cor_r05g_bmv[3][1], cor_r05g_bmv[3][2], cor_r05g_bmv[3][3],
                                  cor_r05g_bmv[3][4], cor_r05g_bmv[3][5], cor_r05g_bmv[3][6], bmv_0)
                if display:
                    print 'giant b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 b-v range!'
            return '{:07.2f}  {:07.2f}'.format(teff[0], teff[1])
        else:
            raise TypeError('mode excess the color indices ranges!')

    elif star_type == 'dwarf':
        ktcs = ksmag - 0.014 + 0.034 * (jmag - ksmag)  # 2005ramirezi
        vmktcs = vmag - ktcs
        vmktcs_0 = vmktcs - e_vmktcs
        if mode == 0:
            if (vmktcs_0 >= 0.40) and (vmktcs_0 <= 1.6):
                teff[0] = formula_teff(a96[0][0], a96[0][1], a96[0][2], a96[0][3], a96[0][4],
                                       a96[0][5], vmktcs_0, feh)
                if display:
                    print 'dwarf v-k (alonso1996) eq.7', '{:.0f}'.format(teff[0])
            elif (vmktcs_0 > 1.60) and (vmktcs_0 <= 2.2):
                teff[0] = formula_teff(a96[1][0], a96[1][1], a96[1][2], a96[1][3], a96[1][4],
                                       a96[1][5], vmktcs_0, feh)
                if display:
                    print 'dwarf v-k (alonso1996) eq.8', '{:.0f}'.format(teff[0])
            else:
                print 'out of alonso1996 v-k range!'
            teff_a = formula_teff(r05d[0][0], r05d[0][1], r05d[0][2], r05d[0][3],
                                  r05d[0][4], r05d[0][5], vmk_0, feh)
            if (vmk_0 >= 0.896) & (vmk_0 <= 3.360) and (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_vmk[0][0], cor_r05d_vmk[0][1], cor_r05d_vmk[0][2], cor_r05d_vmk[0][3],
                                  cor_r05d_vmk[0][4], cor_r05d_vmk[0][5], cor_r05d_vmk[0][6], vmk_0)
                if display:
                    print 'dwarf v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.060) & (vmk_0 <= 2.665) and (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_vmk[1][0], cor_r05d_vmk[1][1], cor_r05d_vmk[1][2], cor_r05d_vmk[1][3],
                                  cor_r05d_vmk[1][4], cor_r05d_vmk[1][5], cor_r05d_vmk[1][6], vmk_0)
                if display:
                    print 'dwarf v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.101) & (vmk_0 <= 2.670) and (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_vmk[2][0], cor_r05d_vmk[2][1], cor_r05d_vmk[2][2], cor_r05d_vmk[2][3],
                                  cor_r05d_vmk[2][4], cor_r05d_vmk[2][5], cor_r05d_vmk[2][6], vmk_0)
                if display:
                    print 'dwarf v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmk_0 >= 1.126) & (vmk_0 <= 2.596) and (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_vmk[3][0], cor_r05d_vmk[3][1], cor_r05d_vmk[3][2], cor_r05d_vmk[3][3],
                                  cor_r05d_vmk[3][4], cor_r05d_vmk[3][5], cor_r05d_vmk[3][6], vmk_0)
                if display:
                    print 'dwarf v-k (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 v-k range!'

            if (vmk_0 >= 0.78) & (vmk_0 <= 3.15) & (feh >= -5.0) & (feh <= 0.40):
                teff[2] = formula_teff(c10[0][0], c10[0][1], c10[0][2], c10[0][3], c10[0][4],
                                       c10[0][5], vmk_0, feh)
                if display:
                    print 'dwarf v-k (casagrande2010)', '{:.0f}'.format(teff[2])
            else:
                print 'out of casagrande2010 v-k range!'
            return '{:07.2f}  {:07.2f}  {:07.2f}'.format(teff[0], teff[1], teff[2])

        if mode == 1:
            teff_a = formula_teff(r05d[1][0], r05d[1][1], r05d[1][2], r05d[1][3],
                                  r05d[1][4], r05d[1][5], vmj_0, feh)
            if (vmj_0 >= 0.815) & (vmj_0 <= 2.608) & (feh >= -0.5) & (feh <= 0.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05d_vmj[0][0], cor_r05d_vmj[0][1], cor_r05d_vmj[0][2], cor_r05d_vmj[0][3],
                                  cor_r05d_vmj[0][4], cor_r05d_vmj[0][5], cor_r05d_vmj[0][6], vmj_0)
                if display:
                    print 'dwarf v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 0.860) & (vmj_0 <= 2.087) & (feh >= -1.5) & (feh <= -0.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05d_vmj[1][0], cor_r05d_vmj[1][1], cor_r05d_vmj[1][2], cor_r05d_vmj[1][3],
                                  cor_r05d_vmj[1][4], cor_r05d_vmj[1][5], cor_r05d_vmj[1][6], vmj_0)
                if display:
                    print 'dwarf v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 0.927) & (vmj_0 <= 1.983) & (feh >= -2.5) & (feh <= -1.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05d_vmj[2][0], cor_r05d_vmj[2][1], cor_r05d_vmj[2][2], cor_r05d_vmj[2][3],
                                  cor_r05d_vmj[2][4], cor_r05d_vmj[2][5], cor_r05d_vmj[2][6], vmj_0)
                if display:
                    print 'dwarf v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            elif (vmj_0 >= 0.891) & (vmj_0 <= 1.932) & (feh >= -4.0) & (feh <= -2.5):
                teff[0] = teff_a + \
                          cor_r05(cor_r05d_vmj[3][0], cor_r05d_vmj[3][1], cor_r05d_vmj[3][2], cor_r05d_vmj[3][3],
                                  cor_r05d_vmj[3][4], cor_r05d_vmj[3][5], cor_r05d_vmj[3][6], vmj_0)
                if display:
                    print 'dwarf v-j (ramirez2005)', '{:.0f}'.format(teff[0])
            else:
                print 'out of ramirez2005 v-j range!'

            teff_b = formula_teff(r05d[2][0], r05d[2][1], r05d[2][2], r05d[2][3],
                                  r05d[2][4], r05d[2][5], vmh_0, feh)
            if (vmh_0 >= 0.839) & (vmh_0 <= 3.215) & (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05d_vmh[0][0], cor_r05d_vmh[0][1], cor_r05d_vmh[0][2], cor_r05d_vmh[0][3],
                                  cor_r05d_vmh[0][4], cor_r05d_vmh[0][5], cor_r05d_vmh[0][6], vmh_0)
                if display:
                    print 'dwarf v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.032) & (vmh_0 <= 2.532) & (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05d_vmh[1][0], cor_r05d_vmh[1][1], cor_r05d_vmh[1][2], cor_r05d_vmh[1][3],
                                  cor_r05d_vmh[1][4], cor_r05d_vmh[1][5], cor_r05d_vmh[1][6], vmh_0)
                if display:
                    print 'dwarf v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.070) & (vmh_0 <= 2.535) & (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05d_vmh[2][0], cor_r05d_vmh[2][1], cor_r05d_vmh[2][2], cor_r05d_vmh[2][3],
                                  cor_r05d_vmh[2][4], cor_r05d_vmh[2][5], cor_r05d_vmh[2][6], vmh_0)
                if display:
                    print 'dwarf v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (vmh_0 >= 1.093) & (vmh_0 <= 2.388) & (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_b + \
                          cor_r05(cor_r05d_vmh[3][0], cor_r05d_vmh[3][1], cor_r05d_vmh[3][2], cor_r05d_vmh[3][3],
                                  cor_r05d_vmh[3][4], cor_r05d_vmh[3][5], cor_r05d_vmh[3][6], vmh_0)
                if display:
                    print 'dwarf v-h (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 v-h range!'
            return '{:07.2f}  {:07.2f}'.format(teff[0], teff[1])

        if mode == 2:
            if (bmv_0 >= 0.20) & (bmv_0 <= 0.80):
                teff[0] = formula_teff(a96[2][0], a96[2][1], a96[2][2], a96[2][3], a96[2][4],
                                       a96[2][5], bmv_0, feh)
                if display:
                    print 'dwarf b-v (alonso1996) eq.1', '{:.0f}'.format(teff[0])

            teff_a = formula_teff(r05d[3][0], r05d[3][1], r05d[3][2], r05d[3][3],
                                  r05d[3][4], r05d[3][5], bmv_0, feh)
            if (bmv_0 >= 0.310) & (bmv_0 <= 1.507) & (feh >= -0.5) & (feh <= 0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_bmv[0][0], cor_r05d_bmv[0][1], cor_r05d_bmv[0][2], cor_r05d_bmv[0][3],
                                  cor_r05d_bmv[0][4], cor_r05d_bmv[0][5], cor_r05d_bmv[0][6], bmv_0)
                if display:
                    print 'dwarf b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0.307) & (bmv_0 <= 1.202) & (feh >= -1.5) & (feh <= -0.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_bmv[0][0], cor_r05d_bmv[0][1], cor_r05d_bmv[0][2], cor_r05d_bmv[0][3],
                                  cor_r05d_bmv[0][4], cor_r05d_bmv[0][5], cor_r05d_bmv[0][6], bmv_0)
                if display:
                    print 'dwarf b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0.335) & (bmv_0 <= 1.030) & (feh >= -2.5) & (feh <= -1.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_bmv[0][0], cor_r05d_bmv[0][1], cor_r05d_bmv[0][2], cor_r05d_bmv[0][3],
                                  cor_r05d_bmv[0][4], cor_r05d_bmv[0][5], cor_r05d_bmv[0][6], bmv_0)
                if display:
                    print 'dwarf b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            elif (bmv_0 >= 0.343) & (bmv_0 <= 0.976) & (feh >= -4.0) & (feh <= -2.5):
                teff[1] = teff_a + \
                          cor_r05(cor_r05d_bmv[3][0], cor_r05d_bmv[3][1], cor_r05d_bmv[3][2], cor_r05d_bmv[3][3],
                                  cor_r05d_bmv[3][4], cor_r05d_bmv[3][5], cor_r05d_bmv[3][6], bmv_0)
                if display:
                    print 'dwarf b-v (ramirez2005)', '{:.0f}'.format(teff[1])
            else:
                print 'out of ramirez2005 b-v range!'

            if (bmv_0 >= 0.18) & (bmv_0 <= 1.29) & (feh >= -5.0) & (feh <= 0.40):
                teff[2] = formula_teff(c10[1][0], c10[1][1], c10[1][2], c10[1][3], c10[1][4],
                                       c10[1][5], bmv_0, feh)
                if display:
                    print 'dwarf b-v (casagrande2010)', '{:.0f}'.format(teff[2])
            else:
                print 'out of casagrande12010 b-v range!'
            return '{:07.2f}  {:07.2f} {:07.2f}'.format(teff[0], teff[1], teff[2])

        else:
            raise TypeError('mode excess the color indices ranges!')

#======================================================================================================================
def cal_logg(vmag, a_v, plx, teff, mass, feh=0.0, test=False):
    global mbol_star  # change the global variable
    mbol_sun, logg_sun, teff_sun = 4.77, 4.44, 5777.
    x = np.log10(teff) - 3.52
    if (np.log10(teff) >= 3.50) & (np.log10(teff) <= 3.67) & (feh >= -0.50) & (feh <= 0.20) or \
       (np.log10(teff) >= 3.56) & (np.log10(teff) <= 3.67) & (feh >= -1.50) & (feh <= -0.50) or \
       (np.log10(teff) >= 3.58) & (np.log10(teff) <= 3.67) & (feh >= -2.50) & (feh <= -1.50) or \
       (np.log10(teff) >= 3.61) & (np.log10(teff) <= 3.67) & (feh >= -3.00) & (feh <= -2.50):
        bc = -0.05531 / x - 0.6177 + 4.420 * x - 2.669 * x**2 + 0.6943 * x * feh - 0.1071 * feh - 0.008612 * feh**2
    elif (np.log10(teff) >= 3.65) & (np.log10(teff) <= 3.96) & (feh >= -0.50) & (feh <= 0.20) or \
         (np.log10(teff) >= 3.65) & (np.log10(teff) <= 3.83) & (feh >= -1.50) & (feh <= -0.50) or \
         (np.log10(teff) >= 3.65) & (np.log10(teff) <= 3.80) & (feh >= -2.50) & (feh <= -1.50) or \
         (np.log10(teff) >= 3.65) & (np.log10(teff) <= 3.74) & (feh >= -3.00) & (feh <= -2.50):
        bc = -0.09930 / x + 0.02887 + 2.275 * x - 4.425 * x**2 + 0.3505 * x * feh - 0.05558 * feh - 0.005375 * feh**2
    else:
        raise ValueError('out of ranges of applications!')
    if test:  # 2014Bergemann_A&A_565_89
        print 'The test correction in A_V should be for SFD98'
        e_bv = a_v / 3.1
        if e_bv > 0.1:
            e_bv = 0.035 + 0.65*e_bv
            a_v = 3.1*e_bv
            mbol_star = vmag + bc + 5 * np.log10(plx) + 5.0 - a_v  # plx in arcsec
    else:
        mbol_star = vmag + bc + 5 * np.log10(plx) + 5.0 - a_v  # plx in arcsec
    logg = logg_sun + np.log10(mass) + 4 * np.log10(teff / teff_sun) + 0.4 * (mbol_star - mbol_sun)
    return '{:.3f}'.format(logg)

#======================================================================================================================
def mass_age_giant(teff, logg, mh, cm, nm):
    cn = cm - nm
    cnm = cm + nm
    if mh > -0.8 and 4000. < teff < 5000. and 1.8 < logg < 3.3 and -0.25 < cm < 0.15 and -0.1 < nm < 0.45 and \
    -0.1 < cnm < 0.15 and -0.6 < cn < 0.2:
        mass = 95.87 - 10.40*mh - 0.73*mh**2 + 41.36*cm - 5.32*cm*mh - 46.78*cm**2 + 15.05*nm - 0.93*nm*mh - \
        30.52*nm*cm - 1.61*nm**2 - 67.61*cnm + 7.05*cnm*mh + 133.58*cnm*cm + 38.94*cnm*nm - 88.99*cnm**2 \
        - 144.18*(teff/4000) + 5.12*(teff/4000)*mh - 73.77*(teff/4000)*cm - 15.29*(teff/4000)*nm + \
        101.75*(teff/4000)*cnm + 27.77*(teff/4000)**2 - 9.42*logg + 1.52*logg*mh + 16.04*logg*cm + 1.35*logg*nm \
        - 18.65*logg*cnm + 28.80*logg*(teff/4000) - 4.10*logg**2

        logage = -54.35 + 6.53*mh + 0.74*mh**2 - 19.02*cm + 4.04*cm*mh + 26.90*cm**2 - 12.18*nm + 0.76*nm*mh + \
        13.33*nm*cm - 1.04*nm**2 + 37.22*cnm - 4.94*cnm*mh - 77.84*cnm*cm - 17.60*cnm*nm + 51.24*cnm**2 \
        + 59.58*(teff/4000) - 1.46*(teff/4000)*mh + 48.29*(teff/4000)*cm + 13.99*(teff/4000)*nm - \
        65.67*(teff/4000)*cnm + 15.54*(teff/4000)**2 + 16.14*logg - 1.56*logg*mh - 13.12*logg*cm - \
        1.77*logg*nm + 14.24*logg*cnm - 34.68*logg*(teff/4000) + 4.17*logg**2
    else:
        raise ValueError('Out of ranges of relations!')
    return '{:.3f} {:.3f}'.format(mass, logage)

#======================================================================================================================
def cal_age(teff,dt,logg,dg,feh,dm):
    #Zini, Age, Mini, Mass, logL, logTe, logg
    from ezpadova import parsec
    iso = parsec.get_t_isochrones(logt0, logt1, dlogt, metal)
    teff_iso = iso['logT'][(iso['logT'] >= (iso['logT']+3*dt)) & (iso['logT'] <= (iso['logT'] - 3*dt))]
    logg_iso = iso['logg'][(iso['logg'] >= (iso['logg']+3*dg)) & (iso['logg'] <= (iso['logg'] - 3*dg))]
    feh_iso  = iso['Z'][(iso['Z'] >= (iso['Z']+3*dm)) & (iso['Z'] <= (iso['Z'] - 3*dm))]
    p = np.exp(-(teff - t_iso)**2/(2*dt**2)) * np.exp(-(logg - g_iso)**2/(2*dg**2)) * np.exp(-(feh - m_iso)**2/(2*dm**2))
    dp = np.sum(p)/dlogt
    return

#======================================================================================================================
if __name__ == '__main__':
    # calculate multiple Teff
    data_Teff = pd.read_csv('/Users/zyt/Desktop/Lirich/Lirich_v1.dat', delim_whitespace=True, usecols=range(23),
                            skiprows=1)
    # for  jmag, hmag, ksmag, bmag, vmag, e_bv, feh in data_Teff.Jmag, data_Teff.Hmag, data_Teff.Kmag, \
    #     data_Teff.Bmag, data_Teff.Vmag, data_Teff.E_BV_SF, data_Teff.FeH:
    Teff=[]
    for i in range(len(data_Teff.Tel)):
        Teffi = cal_teff(data_Teff.Jmag[i], data_Teff.Hmag[i], data_Teff.Kmag[i], data_Teff.Bmag[i],
                 data_Teff.Vmag[i], data_Teff.E_BV_SF[i], data_Teff.FeH[i], display=False)
        Teff.append(Teffi)
        print Teffi
    # print Teff