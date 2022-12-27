# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:23:11 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import copy

## The code is the same as in the previous exercises

np.random.seed()

def pbc(x, L):          ## Periodic boundary conditions
    x[(x<-L/2)] += L
    x[(x>L/2)] -= L
    return x
    
def LJ_force(pos, Nparticles, cf, L, epsilon, sigma):      ## Force induced by the Lennard-Jones potential
    
    F = np.zeros((Nparticles,3))
    f = np.zeros(3) 
    for i in range(Nparticles):
        for j in range(i+1, Nparticles):
            rij = pbc(pos[i]-pos[j], L)

            dij = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
            
            if dij<cf:
                
                f = 4*epsilon*(
                (12*sigma**12/(dij**14)) - (6*sigma**6/(dij**8)))*rij
                
                F[i] += f
                
                F[j] -=  f
                    
    return F
      

   
def vel_verlet(pos, v, dt, Nparticles, cf, L, epsilon, sigma, F):     ## Velocity Verlet algorithm
       
    pos = pbc(pos + v*dt + F*dt**2/2, L)    
    Fnew = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)
    v += (F+Fnew)*dt/2    
    F = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)    
    
    return pos, v, F

    




def therm_Andersen(v, nu, sigma_andersen):   ## Andersen thermostat algorithm
    np.random.seed()
    for n in range(len(v)):
        
        if np.random.rand() < nu:
            v[n] = np.random.normal(0,sigma_andersen, 3)
            
    return v


def kinetic(v):         ## Kinetic energy of the system
        
    return (1./2.)*np.sum((v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2))



Nparticles = 125
Nsteps = 10000
dt = 0.0001

rho = 0.8  ## Densidad de partículas
L = (Nparticles/rho)**(1./3.)  ## Longitud de la caja
epsilon = 1. ## Epsilon para el potencial de lennard Jones
sigma = 1. ## Sigma para el potencial de Lennard Jones
cf = L/2. ## cutoff
M = int(Nparticles**(1./3.))  ## Número de partículas por lado
a = L/M  ## Distancia entre partículas
v = np.zeros((Nparticles, 3))  ## Array inicial de las velocidades
nu = 0.1 ## Probabilidad de aceptación del thermal Andersen
Temp = 2. ## Temperatura

sigma_andersen = np.sqrt(Temp)  ## Sigma Andersen


pos = [] ## Array de las posiciones


for nx in range(M):         ##  Posiciones iniciales (cúbica simple)
    for ny in range(M):
        for nz in range(M):
            pos.append([nx, ny, nz])
            
            
pos = np.array(pos)*a            
F = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)

    
 
    
for step in range(Nsteps):     ## Empieza la simulación    
    

    pos, v, F = vel_verlet(pos, v, dt, Nparticles, cf, L, epsilon, sigma, F)       
    v = therm_Andersen(v, nu, sigma_andersen)      

 
 ## We plot the final distribution
  
plt.figure(figsize=(8,6))
plt.title('Radial distribution', fontsize = 16)
plt.hist(pos[:,0]*3.4, bins=90, color = 'darkgoldenrod')
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.xlim(0, (L/2)*3.4)
plt.xlabel('Distance to the center/(Å)', fontsize = 14)
    
file = open("C:/Users/Usuario/Desktop/Apuntes/Master/MoMo/Carles/Pfinal/Distribution.txt", "w+")
    
for i in range(Nparticles):
    file.write(str(pos[i, 0]) + ' ' + str(pos[i, 1]) + ' ' + str(pos[i, 2]) + '\n')
    
file.close()  
    