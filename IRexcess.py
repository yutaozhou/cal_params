#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#====================
# @File: name.py  
# @Author: ZYT  
# @Time: 2018/07/17
#====================
import numpy as np
import pandas as pd
# import plot_config
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
names = ['photometry','ALi']
rebull2015 = pd.read_csv('/Users/zyt/paper/zyt/paper1/rebull.dat', delim_whitespace=True, skiprows=1, usecols=[0,1], names = names)
excess = [70, 74, 76, 78, 79, 80, 81, 82, 83, 84, 85]
unexcess = rebull2015.drop(rebull2015.index[excess])
excess   = rebull2015.iloc[excess]
efp = FP('Times New Roman', size=22)
fig, ax = plt.subplots(figsize=(10,9))

ax.scatter(unexcess.photometry, unexcess.ALi, c='k', s=80)
ax.scatter(excess.photometry, excess.ALi, c='blue', s=80)
ax.scatter(1.734,5.2, c='r', marker='*',label='Hyper Li target', s=300)

ax.set_xlabel('[3.4] - [22]', fontsize=23)
ax.set_ylabel('A(Li)$_\mathrm{NLTE}$',fontsize=23)
ax.set_xlim([-2, 10])
ax.set_ylim([-2, 5.5])
ax.plot([0,0],[-2,6.0], c='lightgray', alpha=0.75, linewidth=4, zorder=0)
ax.tick_params(axis='both', which='major', labelsize=18.5)
# plt.show()
plt.savefig('/Users/zyt/Desktop/figure_excess.pdf')