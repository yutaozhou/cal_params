#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#====================
# @File: name.py  
# @Author: ZYT  
# @Time: 2018/07/17
#====================
import pandas as pd
import numpy as np
from params import cal_teff
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

# ------------------------ python code begins here
# name the output file
pdfname = '/Users/zyt/Desktop/comp_teff.pdf'

# read input data from two given files
df = pd.read_csv('/Users/zyt/Data/Kepler_para&model_v2/Kepler_para_v2.dat', skiprows=4,
                 delim_whitespace=True)
df['FeH2'] = pd.to_numeric(df['FeH2'], errors='coerce')
df['logg2'] = pd.to_numeric(df['logg2'], errors='coerce')
result = []
for i, logg, Jmag, Hmag, Kmag, Bmag, Vmag, E_BV, feh in zip(range(len(df)), df.logg2, df.Jmag, df.Hmag, df.Kmag,
                                                            df.Bmag, df.Vmag, df.E_BV, df.FeH2):
	if np.isnan(logg):
		logg = 0.0  # as a giant to calculate teff
	if logg >= 3.5:
		tmp = cal_teff(Jmag, Hmag, Kmag, Bmag, Vmag, E_BV, feh, star_type='dwarf', display=False)
	else:
		tmp = cal_teff(Jmag, Hmag, Kmag, Bmag, Vmag, E_BV, feh, star_type='giant', display=False)
	print i, '-->', tmp.split()[0]
	if float(tmp.split()[0]) == 0.0:
		result.append(float(tmp.split()[1]))  # ramirez2005
	elif float(tmp.split()[0]) < 0.0:
		result.append(np.nan)
	else:
		result.append(float(tmp.split()[0]))  # alonso199x
teff_irfm = np.array(result)
teff_spec = np.array(pd.to_numeric(df['Teff2'], errors='coerce'))
delta_teff = teff_spec - teff_irfm

# create the main figure
fig = plt.figure(figsize=(8, 6), dpi=120)
fig.subplots_adjust(hspace=0.05, wspace=0.0001, bottom=0.1, top=0.96, left=0.2, right=1.5)

# define two subplots with different sizes
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])

# upper plot
upperplot = plt.subplot(gs[0])
for logg, teff_s, teff_i in zip(df['logg2'], teff_spec, teff_irfm):
	if logg > 3.5:
		dwarf=plt.scatter(teff_s, teff_i, c='blue')
	else:
		giant=plt.scatter(teff_s, teff_i, c='red')
# errorbar(phase, rv, err, fmt='ko')
plot([3000, 6000], [3000, 6000], 'k-')
ylabel("Teff_IRFM (K)")
xlim([3000, 6000])
ylim([3000, 6000])

plt.legend([dwarf, giant], ["dwarf", "giant"])
plt.setp(upperplot.get_xticklabels(), visible=False)

# lower plot
lowerplot = plt.subplot(gs[2])
for logg, teff_s, teff_i in zip(df['logg2'], teff_spec, delta_teff):
	if logg > 3.5:
		plt.scatter(teff_s, teff_i, c='blue')
	else:
		plt.scatter(teff_s, teff_i, c='red')
# errorbar(phase, oc, err, fmt='ko')
plot([3000, 6000], [0., 0.], 'k:')
xlabel("Teff_spec (K)")
ylabel("delta_teff")
xlim([3000, 6000])
# ylim([3000, 6000])

plt.savefig(pdfname, bbox_inches='tight')
# ------------------------ python code ends here