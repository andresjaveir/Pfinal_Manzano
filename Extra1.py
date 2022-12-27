# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 14:59:15 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import os




## We load the MSD data

HOME = 'C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/Apartado2'
DATA = os.path.join(HOME, 'MSD.txt')

MSD = np.loadtxt(DATA)
MSD005 = MSD[:, 0]*3.4**2 ## MSD in Angstroms
MSD08 = MSD[:, 1]*3.4**2

epsilon = 0.998 ## Lennard-Jones parameter in kJ/mol
m = (40/(1000*6.022e23)) ## Mass in units of kg
dt = 0.0001*np.sqrt(m/epsilon)*10e12 ## We calculate time in units of pico-seconds

t = np.linspace(dt, 100000*dt, 100000) ## The time

D = np.array([MSD005[99999]/(6*100000*0.0001), MSD08[99999]/(6*100000*0.0001)])
rho = np.array([0.05, 0.8]) ## The array of densities in reduced units
densidades = (64*40/(6.022e23))/((64/rho)*(3.4e-8)**3) ## The array of densities 
                                                        # in units of g/cm^3

plt.figure(figsize=(8,6))  ## We plot MSD
plt.title('Mean-square displacement', fontsize = 16)
plt.plot(t, MSD005, label='$\u03C1=0.084 g/cm^3$', color = 'lime')
plt.plot(t, MSD08, label='$\u03C1=1.35 g/cm^3$', color = 'g') 
plt.xlabel('Time/(ps)', fontsize = 14)
plt.ylabel('Mean-Square displacement/($Å^2$)', fontsize = 14)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left')




