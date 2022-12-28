# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 00:35:05 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import copy

## This code is very similar to the one used in exercise1, there are a 
#  few changes that will be outlined.

np.random.seed()



def pbc(x, L):          ## Periodic boundary conditions
    x[(x<-L/2)] += L
    x[(x>L/2)] -= L
    return x
    
def LJ_force(pos, Nparticles, cf, L, epsilon, sigma):      ## Force induced by the Lennard-Jones potential
                                                        # same as in the previous exercise
    F = np.zeros((Nparticles,3))
    Upot = 0
    p = 0
    f = np.zeros(3)
    
    for i in range(Nparticles):
        for j in range(i+1, Nparticles):
            rij = pbc(pos[i]-pos[j], L)

            dij = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
            
            if dij<cf:
            
                f =     4*epsilon*(
                (12*sigma**12/(dij**14)) - (6*sigma**6/(dij**8)))*rij
            
                F[i] += f
                
                F[j] -=  f
                
                Upot += 4*epsilon*((sigma/dij)**12 - (sigma/dij)**6) - (
                    4*epsilon*((sigma/cf)**12 - (sigma/cf)**6))
                
                ## The second part of the pressure indicated by the Virial theorem
                
                p += np.abs(np.sum(f*rij)) ## This is different respect to the
                                # previous exercise.
                                
                    
    return F, Upot, p

def vel_verlet(pos, v, dt, Nparticles, cf, L, epsilon, sigma, F):     ## Velocity Verlet algorithm
       
    pos = pbc(pos + v*dt + F*dt**2/2, L)    
    Fnew, Upot, p = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)
    v += (F+Fnew)*dt/2    
    F = Fnew   
    
    return pos, v, F, Upot, p
      



def therm_Andersen(v, nu, sigma_andersen):   ## Andersen thermostat algorithm
    np.random.seed()
    for n in range(len(v)):
        
        if np.random.rand() < nu:
            
            v[n] = np.random.normal(0,sigma_andersen, 3)
            
    return v


def kinetic(v):         ## Kinetic energy of the system
        
    return (1./2.)*np.sum((v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2))


Nparticles = 64 ##Number of particles
rho = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8]) ## Density in reduced units
Temp = 1.2 ## Temperature in reduced units
sigma_andersen = np.sqrt(Temp) ## Andersen thermostat parameter
L = (Nparticles/rho)**(1./3.)  ## Box's side in reduced units
epsilon = 1. ## Lennard-Jones parameter in reduced units
sigma = 1. ## Lennard-Jones parameter in reduce units
cf = L/2. ## cutoff distance
M = int(Nparticles**(1./3.))+1  ## Number of particles per row
a = L/M  ## Distance between particles
v = np.zeros((Nparticles, 3))  ## Initial array for the velocities
nu = 0.1 ## Andresen probability of acceptance
Nsteps = 100000 ## Number of steps
dt = 0.0001 ## Time step in reduced units
rinicial = [] ## initial array for the positions


for nx in range(M):         ##  Initial system (simple cubic)
    for ny in range(M):
        for nz in range(M):
            rinicial.append([nx, ny, nz])
            
            
rinicial = np.array(rinicial) ## Initial configuration

r = np.zeros((len(rho), Nparticles, 3))  ## The array for my initial system

for i in range(len(rho)): ## We set the system
    
    r[i] = rinicial*a[i]
        

v = np.zeros((len(rho), Nparticles, 3)) ## The array for the velocities
F = np.zeros((len(rho), Nparticles, 3)) ## The array for the forces

rinicial = np.zeros((2, Nparticles, 3))
rinicial[0] = r[0] ## We need to save the configuration for the first
rinicial[1] = r[5] # and the last systems, to calculate the MSD

pot = np.zeros((len(rho), Nsteps)) ## The array for the potentials
k = np.zeros((len(rho), Nsteps)) ## The array for the kinetic energies
P = np.zeros((len(rho), Nsteps)) ## The array for the pressures

MSD = np.zeros((2, Nsteps))  ## The array for the MSD


## We open the files, to be able to save positions to see the trajectories later

filepos = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones.txt", "w+")
filepos2 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones2.txt", "w+")
filepos3 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones3.txt", "w+")
filepos4 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones4.txt", "w+")
filepos5 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones5.txt", "w+")
filepos6 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/posiciones6.txt", "w+")


for i in range(Nsteps):
    
    ## We calculate the position, the velocity, the force, the 
    # potential energy and the pressure using the Velocity Verlet
    # algorithm and the Lennard Jones interaction function.
    
    r[0], v[0], F[0], pot[0, i], P[0, i] = vel_verlet(r[0], v[0], dt, Nparticles,
                                            cf[0], L[0], epsilon, sigma, F[0])
    r[1], v[1], F[1], pot[1, i], P[1, i] = vel_verlet(r[1], v[1], dt, Nparticles, 
                                             cf[1], L[1], epsilon, sigma, F[1])
    r[2], v[2], F[2], pot[2, i], P[2, i] = vel_verlet(r[2], v[2], dt, Nparticles, 
                                             cf[2], L[2], epsilon, sigma, F[2])
    r[3], v[3], F[3], pot[3, i], P[3, i] = vel_verlet(r[3], v[3], dt, Nparticles, 
                                             cf[3], L[3], epsilon, sigma, F[3])
    r[4], v[4], F[4], pot[4, i], P[4, i] = vel_verlet(r[4], v[4], dt, Nparticles, 
                                             cf[4], L[4], epsilon, sigma, F[4])
    r[5], v[5], F[5], pot[5, i], P[5, i] = vel_verlet(r[5], v[5], dt, Nparticles, 
                                             cf[5], L[5], epsilon, sigma, F[5])
    ## We change the velocities with Andersen thermostat, to simulate
    # a thermal bath
    
    v[0] = therm_Andersen(v[0], nu, sigma_andersen)
    v[1] = therm_Andersen(v[1], nu, sigma_andersen)
    v[2] = therm_Andersen(v[2], nu, sigma_andersen)
    v[3] = therm_Andersen(v[3], nu, sigma_andersen)
    v[4] = therm_Andersen(v[4], nu, sigma_andersen)
    v[5] = therm_Andersen(v[5], nu, sigma_andersen)
        
    ## We calculate kinetic energy
    
    k[0][i] = kinetic(v[0])
    k[1][i] = kinetic(v[1])
    k[2][i] = kinetic(v[2])
    k[3][i] = kinetic(v[3])
    k[4][i] = kinetic(v[4])
    k[5][i] = kinetic(v[5])
    
    ##  We measure the MSD  
    MSD[0, i] = (1/Nparticles)*np.sum((pbc(r[0]-rinicial[0], L[0])**2)) ## We apply the pbc to the MSD, in case a particle jumps it doesn't calculate
    MSD[1, i] = (1/Nparticles)*np.sum((pbc(r[5]-rinicial[1], L[5])**2)) # a wrong distance for the MSD
    
    
    if i%100 == 0: ## We save the positions of the particles every 100 iterations
    
        
        for j in range(Nparticles):
            for l in range(3):
                filepos.write(str(r[0, j, l]))
                filepos.write(' ')
                
                filepos2.write(str(r[1, j, l]))
                filepos2.write(' ')
                
                filepos3.write(str(r[2, j, l]))
                filepos3.write(' ')
                
                filepos4.write(str(r[3, j, l]))
                filepos4.write(' ')
                
                filepos5.write(str(r[4, j, l]))
                filepos5.write(' ')
                
                filepos6.write(str(r[5, j, l]))
                filepos6.write(' ')
            
            filepos.write('\n')
            filepos2.write('\n')
            filepos3.write('\n')
            filepos4.write('\n')
            filepos5.write('\n')
            filepos6.write('\n')


filepos.close()
filepos2.close()
filepos3.close()
filepos4.close()
filepos5.close()
filepos6.close()

tmax = dt*Nsteps

## The mean square displacement
D = np.array([MSD[0,Nparticles-1]/(6*tmax), MSD[1,Nparticles-1]/(6*tmax)])

## We save the arrays for the MSD, the kinetic energy, the potential energy
# and the pressure

file = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/Upot.txt", "w+")
file2 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/Kin.txt", "w+")
file3 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/MSD.txt", "w+")
file4 = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/presion.txt", "w+")


for i in range(Nsteps):
    
    file3.write(str(MSD[0,i]))
    file3.write(' ')
    
    file3.write(str(MSD[1, i]))
    file3.write('\n')

    
    for j in range(len(rho)):
        
        file.write(str(pot[j, i]))
        file.write(' ')
        
        file2.write(str(k[j, i]))
        file2.write(' ')
        
        file4.write(str(P[j, i]))
        file4.write(' ')
        
    file.write('\n')
    file2.write('\n')
    file4.write('\n')
    
file.close()
file2.close()
file3.close()
file4.close()




