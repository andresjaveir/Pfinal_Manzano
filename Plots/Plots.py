# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 11:45:50 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import os


## First we load the data of the kinetic energy, the potential energy and 
# the pressure

HOME = 'C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/Apartado2'
DATAK = os.path.join(HOME, 'Kin.txt')
DATAU = os.path.join(HOME, 'Upot.txt')  


kin = np.loadtxt(DATAK)
U = np.loadtxt(DATAU)

## We define the variables for the sides of the boxes in reduced units

L08 = (64/0.8)**(1/3) ## For rho=0.8
L06 = (64/0.6)**(1/3) ## For rho=0.6
L04 = (64/0.4)**(1/3) ## For rho=0.4
L02 = (64/0.2)**(1/3) ## For rho=0.2
L01 = (64/0.2)**(1/3) ## For rho=0.1
L005 = (64/0.1)**(1/3) ## For rho=0.05

epsilon = 0.998 ## Lennard-Jones parameter in kJ/mol
m = 40 ## Molar mass of Argon in g/mol

sigma = 3.4e-8 ## Lennard-Jones parameter in cm
Na = 6.022e23 ## Avogadro's number

Nparticles = 64 ## Number of particles



kin08 = kin[:,5]*epsilon ## Kinetic energy for rho=0.8 in units of kJ/mol
kin06 = kin[:,4]*epsilon ## Kinetic energy for rho=0.6 in units of kJ/mol
kin04 = kin[:,3]*epsilon ## Kinetic energy for rho=0.4 in units of kJ/mol
kin02 = kin[:,2]*epsilon ## Kinetic energy for rho=0.2 in units of kJ/mol
kin01 = kin[:,1]*epsilon ## Kinetic energy for rho=0.1 in units of kJ/mol
kin005 = kin[:,0]*epsilon ## Kinetic energy for rho=0.05 in units of kJ/mol

U08 = U[:,5]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol
U06 = U[:,4]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol
U04 = U[:,3]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol
U02 = U[:,2]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol
U01 = U[:,1]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol
U005 = U[:,0]*epsilon ## Potential energy for rho=0.8 in units of kJ/mol

T08 = kin08*2/(3*64-3) ## Thermal energy (kbT) for rho=0.8 in units of kJ/mol
T06 = kin06*2/(3*64-3) ## Thermal energy (kbT) for rho=0.6 in units of kJ/mol
T04 = kin04*2/(3*64-3) ## Thermal energy (kbT) for rho=0.4 in units of kJ/mol
T02 = kin02*2/(3*64-3) ## Thermal energy (kbT) for rho=0.2 in units of kJ/mol
T01 = kin01*2/(3*64-3) ## Thermal energy (kbT) for rho=0.1 in units of kJ/mol
T005 = kin005*2/(3*64-3) ## Thermal energy (kbT) for rho=0.05 in units of kJ/mol



rho = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8]) ## Array for the densities
                                                # in reduced units
                                                
                                                
densidades = (Nparticles*m/(Na))/((Nparticles/rho)*(sigma)**3) ## Array for
                                    # the densities in units of g/cm^3
                                               
                                                
## We take the mean value of the energies from the point the system reaches
# equilibrium
              
## Mean values of the equilibrium potential for each system
                                  
potenciales = np.array([np.mean(U005[80000 :]), np.mean(U01[70000 :]), 
                        np.mean(U02[50000 :]), np.mean(U04[20000 :]),
                        np.mean(U06[10000 :]), np.mean(U08[5000 :])])*epsilon


## Mean values of the equilibrium kinetic energy for each system

cineticas = np.array([np.mean(kin005[80000 :]), np.mean(kin01[70000 :]), 
                        np.mean(kin02[50000 :]), np.mean(kin04[20000 :]),
                        np.mean(kin06[10000 :]), np.mean(kin08[5000 :])])

## Mean values of the equilibrium total energy for each system

energias = np.array([cineticas[0]+potenciales[0], cineticas[1]+potenciales[1],
                     cineticas[2]+potenciales[2], cineticas[3]+potenciales[3],
                     cineticas[4]+potenciales[4], 
                     cineticas[5]+potenciales[5]])


## The errors in the measures are given by the standard deviation

errcin = np.array([np.std(kin005[80000 :]), np.std(kin01[70000 :]), ## Kinetic
                        np.std(kin02[50000 :]), np.std(kin04[20000 :]),
                        np.std(kin06[10000 :]), np.std(kin08[5000 :])])



errpot = np.array([np.std(U005[80000 :]), np.std(U01[70000 :]),  ## Potential
                        np.std(U02[50000 :]), np.std(U04[20000 :]),
                        np.std(U06[10000 :]), np.std(U08[5000 :])])

erren = errpot+errcin ## Total energy 


## We plot the results

plt.figure(figsize=(8,6))  ## Total energy
plt.title('Total energy as a function of density', fontsize = 16)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.errorbar(densidades, energias, yerr = erren, fmt = '--or')
plt.xlabel('Density/(g/$cm^3$)', fontsize = 14)
plt.ylabel('Total energy/(kJ/mol)', fontsize = 14)


plt.figure(figsize=(8,6)) ## Kinetic energy
plt.title('Kinetic energy as a function of density', fontsize = 16)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.errorbar(densidades, cineticas, yerr = errcin, fmt = '--or')
plt.xlabel('Density/(g/$cm^3$)', fontsize = 14)
plt.ylabel('Kinetic energy/(kJ/mol)', fontsize = 14)

plt.figure(figsize=(8,6)) ## Potential energy
plt.title('Potential energy as a function of density', fontsize = 16)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.errorbar(densidades, potenciales, yerr = errpot, fmt = '--or')
plt.xlabel('Density/(g/$cm^3$)', fontsize = 14)
plt.ylabel('Potential energy/(kJ/mol)', fontsize = 14)






## Now for the pressure we need different units, so we will make it next
## New units


HOME = 'C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal'
DATAP = os.path.join(HOME, 'presion.txt')
DATAK = os.path.join(HOME, 'Kin_presion.txt')
Presion = np.loadtxt(DATAP)
Kin = np.loadtxt(DATAK)
L = (Nparticles/rho)**(1/3) ## Array of the box' sides in reduced units
Na = 6.022e23 ## Avogadro's number
m = 40/1000 ## Molar mass in kg/mol
M = 40 ## Molar mass in g/mol
epsilon = 0.998*1000*(Nparticles/(Na)) ## Lennard-Jones parameter in J 
sigma = 3.4e-10 ## Lennard-Jones parameter in m
sigma2 = 3.4e-8 ## Lennard-Jones parameter in cm
d = (Nparticles*m/((Na)))/((Nparticles/rho)*(sigma)**3) ## Density in kg/m^3
L = L*sigma ## Array of the box' sides in m
sigma = 3.4e-10 ## Lennard-Jones parameter in m


d1 = (Nparticles*M/((Na)))/((64/rho)*(sigma2)**3) ## Densities in g/cm^3


P = np.array([Presion[:, 0], Presion[:, 1], Presion[:, 2], Presion[:, 3],
             Presion[:, 4], Presion[:, 5]]) ## Array for the second term of the
                                            # Virial theorem
k = np.array([Kin[:, 0], Kin[:, 1], Kin[:, 2], Kin[:, 3],
             Kin[:, 4], Kin[:, 5]]) ## Array for the kinetic energies in reduced 
                                    # units

Temperatura = 2/(3*Nparticles-3)*k  ## Array for the temperature in reduced i


## We take the mean and the error of the pressure in equilibrium.
  
MeanP = np.array([np.mean(P[0][20000 :]), np.mean(P[1][20000 :]), ## Equilibrium
                  np.mean(P[2][20000 :]), np.mean(P[3][20000 :]), # mean of P
                  np.mean(P[4][20000 :]), np.mean(P[5][20000 :])]) # in reduced
                                                                  # units

ErrP = np.array([np.std(P[0][20000 :]), np.std(P[1][20000 :]),  ## Error 
                                               np.std(P[2][20000 :]), # of P
                        np.std(P[3][20000 :]), np.std(P[4][20000 :]), # in Pa
                        np.std(P[5][20000 :])])*epsilon*1/(3*L**3) 


MeanK = np.array([np.mean(k[0][20000 :]), np.mean(k[1][20000 :]),  ## Mean of 
                  np.mean(k[2][20000 :]), np.mean(k[3][20000 :]), # the kinetic
                  np.mean(k[4][20000 :]), np.mean(k[5][20000 :])]) # in reduced
                                                                    # units
                                                                    
MeanT = 2/(3*Nparticles-3)*MeanK ## Mean of the temperature in reduced units


Presion = d*MeanT*epsilon + 1/(3*L**3)*MeanP*epsilon ## The total pressure in
                                                    # Pa

plt.figure(figsize=(8,6)) ## We plot it
plt.errorbar(d1, Presion, yerr = ErrP, fmt = '--or')
plt.xlabel('Density/(g/$cm^3$)', fontsize = 14)
plt.ylabel('Pressure/(Pa)', fontsize = 14)
plt.title('Pressure vs density', fontsize = 16)
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)




