# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 19:22:51 2022

@author: Usuario
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import copy

np.random.seed()

def pbc(x, L):          ## Periodic boundary conditions
    x[(x<-L/2)] += L   ## If the component is smaller than -L/2, L is added
    x[(x>L/2)] -= L   ## If the component is larger than L/2, L is substracted
    return x
    
def LJ_force(pos, Nparticles, cf, L, epsilon, sigma):      ## Force induced by the Lennard-Jones potential
    
    f = np.zeros(3)    ## Auxiliar array to speed up the process
    F = np.zeros((Nparticles,3))  ## Array for the total force acting on the system
    Upot = 0  ## We define the potential energy
    
    for i in range(Nparticles):  ## We begin the loop
        for j in range(i+1, Nparticles):  ## We calculate the interaction of each
                                            # particle j>i over the particle i
                                            # We only consider j>i due to symmetry
                                            # to speed up the calculations
            rij = pbc(pos[i]-pos[j], L) ## Distance between the 2 particles, with pbc

            dij = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2) ## Modulus of the distance
            
            if dij<cf:  ## We take into account the cutoff frequency
                
                f = 4*epsilon*(    ##The vector of the force
                (12*sigma**12/(dij**14)) - (6*sigma**6/(dij**8)))*rij
            
                F[i] += f ## Force of j on i
                
                F[j] -= f  ## Force of i on j
                 
                Upot += 4*epsilon*((sigma/dij)**12 - (sigma/dij)**6) - (
                    4*epsilon*((sigma/cf)**12 - (sigma/cf)**6)) ## The potential energy
                                   
    return F, Upot
      

   
def vel_verlet(pos, v, dt, Nparticles, cf, L, epsilon, sigma, F):     ## Velocity Verlet algorithm
       
    pos = pbc(pos + v*dt + F*dt**2/2, L)  ## The position according to the algorithm  
    Fnew, Upot = LJ_force(pos, Nparticles, cf, L, epsilon, sigma) ## We need the 
                                # force in the new position as well as the previous one to 
                                # properly calculate the velocity
    v += (F+Fnew)*dt/2      ## The evolution of the velocity, using both forces
    F, Upot = LJ_force(pos, Nparticles, cf, L, epsilon, sigma) ## We recalculate the force   
    
    return pos, v, F, Upot

    




def therm_Andersen(v, nu, sigma_andersen):   ## Andersen thermostat algorithm
    np.random.seed()             # to be able to set a temperature to the system
    for n in range(len(v)):  ## Nu is the probability of acceptance
        
        if np.random.rand() < nu: ## We accept the new velocity with probability nu
            v[n] = np.random.normal(0,sigma_andersen, 3) ## Normally distributed
                                                        # velocities           
    return v


def kinetic(v):         ## Kinetic energy of the system
        
    return (1./2.)*np.sum((v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2))


def time_step_Euler_pbc(pos, Nparticles, cf, v, L, epsilon, sigma, dt): ## Time-step
    F, Upot = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)   # Euler algorithm
    pos = pbc(pos + v*dt + 0.5*F*dt**2,L)
    v += F*dt
    return pos, v, F, Upot

## We now begin the initial simulation, to have a system compatible with kbT = 1.2e



Nparticles = 64  ## Number of particles
Nsteps = 10000  ## Number of steps of the initial simulation, to have a system
                # compatible with kbT = 1.2e
dt = 0.0001   ## Time step of the initial simulation

rho = 0.7  ## Particle density
L = (Nparticles/rho)**(1./3.)  ## Side of the box
epsilon = 1. ## Epsilon for the Lennard-Jones potential (reduced units)
sigma = 1. ## Sigma for the Lennard-Jones potential (reduced units)
cf = L/2. ## cutoff value, since it is L/2, particles inside the box
            # will interact with all the other particles
M = int(Nparticles**(1./3.))+1  ## Number of particles per row
a = L/M  ## Distance between particles
v = np.zeros((Nparticles, 3))  ## Initial array for the velocities
nu = 0.1 ## Thermal Andersen probability acceptance
Temp = 100. ## Temperature

sigma_andersen = np.sqrt(Temp)  ## Sigma Andersen


pos = [] ## Array for the positions.

## We now set a random initial configuration

for nx in range(M):         ## Initial positions (simple cubic)
    for ny in range(M):
        for nz in range(M):
            pos.append([nx, ny, nz])
            
            
pos = np.array(pos)*a       ## The initial configuration     
F, Upot = LJ_force(pos, Nparticles, cf, L, epsilon, sigma)


## We now begin the simulation to reach a configuration with kbT = 1.2e

for step in range(Nsteps):      
    

    pos, v, F, Upot = vel_verlet(pos, v, dt, Nparticles, cf, L, epsilon, sigma, F)       
    v = therm_Andersen(v, nu, sigma_andersen)      
    kin = kinetic(v)

## We plot its velocity distribution.

vdist = np.sqrt(v[0, :]**2 + v[1, :]**2 + v[2,:]**2)


plt.figure(figsize=(8,6))
plt.hist(vdist, bins=15, color='darkgoldenrod')
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Before equilibrium')
plt.xlabel('velocity')

## Now that we have the configuration, we copy it, to simulate its evolution
# for different algorithms and values of the time step dt.
dt = np.array([0.0001, 0.001, 0.01, 0.0001])  ## Different timesteps

Nstepslong = 100000   ## Number of steps 
p = np.zeros((len(dt), Nstepslong + 1))  ## The array for the momentum
E = np.zeros((len(dt), Nstepslong + 1)) ## The array for the energy


for i in range(len(dt)):  ## We set the initial value for the array of the
    p[i,0] = np.sum(v)   # energies and the array of the momenta
    E[i,0] = kin+Upot

r = np.zeros((len(dt), Nparticles, 3))  ## We set the array for the positions of each simulation
vel = np.zeros((len(dt), Nparticles, 3)) ## We set the array for the velocities of each simulation
Force = np.zeros((len(dt), Nparticles, 3)) ## We set the array for the forces of each simulation

kin = np.zeros(len(dt)) ## We define the array for the kinetic energies
U = np.zeros(len(dt)) ## We define the array for the potential energies

for i in range(len(dt)):
    r[i] = pos  ## We copy the initial configuration to all my systems
    vel[i] = v
    Force[i] = F
    
v = vel
F = Force

## We now begin the simulation

for step in range(Nstepslong):
    ## We calculate the position, velocity, force and potential using the Velocity Verlet
    #  and the Force LJ functions, the kinetic energy using the kin function, 
    # and the linear momentum, to check it is constant
    r[0], v[0], F[0], U[0] = vel_verlet(r[0], v[0], dt[0], Nparticles, cf, L, epsilon, sigma, F[0])
    kin[0] = kinetic(v[0])
    E[0,step+1] = kin[0] + U[0]
    p[0,step+1] = np.sum(v[0])
    
    r[1], v[1], F[1], U[1] = vel_verlet(r[1], v[1], dt[1], Nparticles, cf, L, epsilon, sigma, F[1])
    kin[1] = kinetic(v[1])
    E[1,step+1] = kin[1] + U[1]
    p[1,step+1] = np.sum(v[1])
    
    r[2], v[2], F[2], U[2] = vel_verlet(r[2], v[2], dt[2], Nparticles, cf, L, epsilon, sigma, F[2])
    kin[2] = kinetic(v[2])
    E[2,step+1] = kin[2] + U[2]
    p[2,step+1] = np.sum(v[2])
    
    
    r[3], v[3], F[3], U[3] = vel_verlet(r[3], v[3], dt[0], Nparticles, cf, L, epsilon, sigma, F[3])
    kin[3] = kinetic(v[3])
    E[3,step+1] = kin[3] + U[3]
    p[3,step+1] = np.sum(v[3])


## Now we plot the values of E and p for the different time steps
    
t = np.zeros((len(dt), Nstepslong+1)) ## We set the array of times

for i in range(len(dt)):
    
    t[i] = np.linspace(0, Nstepslong*dt[i], Nstepslong+1) ## The arrays of times



for i in range(len(dt)): ## We make the plots
    plt.figure(figsize=(8,6))
    plt.plot(t[i], E[i], color='g') ## The energies
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    plt.title('Energy for dt=%f' %dt[i])
    
    plt.figure(figsize=(8,6))
    plt.plot(t[i], p[i], color='g')  ## The momenta
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    plt.title('Momentum for dt=%f' %dt[i])
    
## And now the velocity distributions in comparison with the Maxwell Boltzmann
## We calculate the temperature of each distribution using the equipartition theorem

T = np.array([kin[0]*2/(3*Nparticles-3), kin[1]*2/(3*Nparticles-3), 
              kin[2]*2/(3*Nparticles-3), kin[3]*2/(3*Nparticles-3)]) 
                                                    

vdist = np.zeros((len(dt), Nparticles))  ## The array for the distributions

for i in range(len(dt)): ## The experimental results
    for j in range(Nparticles):
        vdist[i,j] = np.sqrt(v[i, j, 0]**2 + v[i, j, 1]**2 + v[i, j, 2]**2)
    
## The Maxwell Boltzmann distribution

vboltzmann = np.linspace(0, 40, 500)  
f = np.zeros((len(T), len(vboltzmann)))
for i in range(len(T)):
    f[i] = (1/(2*np.pi)**(3/2))*np.exp(-vboltzmann**2/(2*T[i]))*4*np.pi*vboltzmann**2
    

## Now we plot the velocity distributions

for i in range(len(T)):
    plt.figure(figsize=(8,6))
    plt.title('After equilibrium for dt=%f'%dt[i])
    plt.hist(vdist[i], bins=15, color='darkgoldenrod', label="Experimental results")
    plt.plot(vboltzmann, f[i]/3, color='r', label='Theroy')
    plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
    
