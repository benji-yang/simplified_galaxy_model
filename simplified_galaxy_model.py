# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:23:28 2021

@author: Chang Yee Zhwen
"""

"""
Created on Mon Jun 21 18:36:07 2021
@author: Guo Jing Yang
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FuncAnimation
from matplotlib import animation as animation
from matplotlib import style
from mpl_toolkits import mplot3d
import numpy as np
from astropy import constants as const
import math
from astropy import units as u


class pos:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name
        
G_SI = const.G.value
year = 365*24*60*60
klyr = 1000*const.c.value*year
Msol = 1.988e30
G = G_SI

width_x = 0.08
height = 8
width_y = 0.1

num_central = 50
num_outer = 50

total_mass = 1e11 * Msol
mass_ratio = 0.9999
mass_central = total_mass * mass_ratio
indv_mass= (total_mass*(1-mass_ratio))/(num_central + num_outer)

def positions(n_particles, inner=0.5, outer=1): 
    radius = (np.arange(n_particles) + 1) / n_particles * (outer - inner) + inner
    radius = klyr * radius
    theta = np.ones(n_particles) * np.pi / 2.0
    phi = np.arange(n_particles) / n_particles * 2.0 * np.pi
    rx = radius * np.sin(theta) * np.cos(phi)
    ry = radius * np.sin(theta) * np.sin(phi)
    rz = radius * np.cos(theta)    
    r = pos(rx,ry,rz)
    return r

def circularise(rs, mass_central=mass_central):
    """
    Given a set of particle positions initialise their velocities
    to give circular orbits in xy plane. Set z velocity to zero
    for simplicity.
    Assign total velocity according to v^2/2r = GM/r^2
    i.e. v^2 = 2GM/r 
    """
    # Assume particles have been assigned random positions to particles
    #by rs = positions(n_particles)
    #calculate corresponding r^2 values
    r2 = rs.x**2 + rs.y**2 + rs.z**2 
    r = np.sqrt(r2)   
    #calculate velocities that would give circular orbits
    v2 = 1.5 * G * mass_central / r
    vel = np.sqrt(v2)   
    #calculate (theta, phi) for particle positions to get tangential vector

    theta = np.arccos(rs.z / np.sqrt(r2))
    phi = np.arctan(rs.y / rs.x)

    print(r)
    vx = vel * rs.y / r
    vy = vel * -rs.x / r
    vz = vel * np.zeros(len(vel))
    vs = pos(vx,vy,vz)
    print(vs)

    return vs

def Acc(bodies, body_index): #acceleration
    G = const.G.value
    acceleration = pos(0,0,0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2 + (target_body.location.z - external_body.location.z)**2 +0.1**2
            r = np.sqrt(r)
            constant = G * external_body.mass / r**3
            acceleration.x += constant * (external_body.location.x - target_body.location.x)
            acceleration.y += constant * (external_body.location.y - target_body.location.y)
            acceleration.z += constant * (external_body.location.z - target_body.location.z)
    return acceleration

def potentialenergy(bodies, body_index): 
    G = const.G.value
    pe = 0.0
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2 + (target_body.location.z - external_body.location.z)**2
            r = np.sqrt(r)
            constant = -1 * G * external_body.mass / r
            pe += constant*target_body.mass
    return pe
    
def leap_frog(bodies, time_step): 
    for body_index, target_body in enumerate(bodies):
        acceleration = Acc(bodies, body_index)

        target_body.velocity.x += acceleration.x * time_step/2.0
        target_body.velocity.y += acceleration.y * time_step/2.0
        target_body.velocity.z += acceleration.z * time_step/2.0

    for target_body in bodies:
        target_body.location.x += target_body.velocity.x * time_step
        target_body.location.y += target_body.velocity.y * time_step
        target_body.location.z += target_body.velocity.z * time_step
        pos_dict[target_body.name]["x"].append(target_body.location.x)
        pos_dict[target_body.name]["y"].append(target_body.location.y)
        pos_dict[target_body.name]["z"].append(target_body.location.z)
    
    for body_index, target_body in enumerate(bodies):
        acceleration = Acc(bodies, body_index)
        target_body_potential = potentialenergy(bodies, body_index)
        pos_dict[target_body.name]["PE"].append(target_body_potential)
        
        target_body.velocity.x += acceleration.x * time_step/2.0
        target_body.velocity.y += acceleration.y * time_step/2.0
        target_body.velocity.z += acceleration.z * time_step/2.0 
        target_body_total_velocity = 0.5 * target_body.mass * (target_body.velocity.x**2 + target_body.velocity.y**2 + target_body.velocity.z**2)
        vel_dict[target_body.name]["x"].append(target_body.velocity.x)
        vel_dict[target_body.name]["y"].append(target_body.velocity.y)
        vel_dict[target_body.name]["z"].append(target_body.velocity.z)
        vel_dict[target_body.name]["KE"].append(target_body_total_velocity)

names = ["center"]
for i in range(num_central + num_outer):
    j = "body " + str(i+1)
    names.append(j)
#names.append("object")
pos_dict = {}
vel_dict = {}
for i in names:
    pos_dict[i] = {"x": [], "y": [], "z": [], "PE": []}
    vel_dict[i] = {"x": [], "y": [], "z": [], "KE": []}

r_in = positions(num_central, inner=0.5, outer=1.09)
v_in = circularise(r_in)
r_out = positions(num_outer, inner=1.1, outer=5)
x2 = r_out.x/width_x
y2 = r_out.y/width_y
z2 = r_out.z/height *  np.zeros(len(r_out.z))
r_out = pos(x2,y2,z2)
v_out = circularise(r_out)

rx = np.append(r_in.x,r_out.x)
ry = np.append(r_in.y,r_out.y)
rz = np.append(r_in.z,r_out.z)
r_total = pos(rx,ry,rz)

vx = np.append(v_in.x,v_out.x)
vy = np.append(v_in.y,v_out.y)
vz = np.append(v_in.z,v_out.z)
v_total = pos(vx,vy,vz)

#store the initial condition into bodies
initial_dict={"center":{"location" : pos(0,0,0), "mass":mass_central, "velocity": pos(0,0,0)}}
for i in range(num_central + num_outer):
    initial_dict[names[i+1]] = {"location":pos(r_total.x[i],r_total.y[i],r_total.z[i]), "mass" : indv_mass, "velocity" : pos(v_total.x[i],v_total.y[i],v_total.z[i])}
initial_dict["object"] = {"location": pos(10*klyr,0,0), "mass": 10*mass_central, "velocity": pos(-10e6,500000,500000)}

bodies = []
for i in names:
    Body = body(location = initial_dict[i]["location"], mass = initial_dict[i]["mass"], velocity = initial_dict[i]["velocity"], name = i)
    bodies.append(Body)

#using leap frog to array of steps for each particle
for i in range(100):
    leap_frog(bodies, time_step =1.0e5*year)
    

#%%
def plot_frame(pos_dict, step = 0, axis=0):
    """
    Given a list of body objects plot their positions as a scatter plot
    """
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(0.7e20, -0.7e20)
    ax.set_ylim(0.7e20, -0.7e20)
    
    for name in pos_dict.keys():
        loc_dict = pos_dict[name]
        if step < len(loc_dict['x']):
            x = loc_dict['x'][step]
            y = loc_dict['y'][step]
            z = loc_dict['z'][step]
        
        if axis == 0:
            ax.plot3D(x, y, '.')
            plt.style.use('classic')
        elif axis == 1:
            ax.plot3D(x, z, '.')
            plt.style.use('classic')
        else:
            ax.plot3D(y, z, '.')
            plt.style.use('classic')
    
def plot_blur(pos_dict, nsteps=30, axis = 0):
    
    for step in range(nsteps):
        print(step)
        plot_frame(pos_dict, step, axis)

plot_blur(pos_dict, nsteps = 100, axis = 0)









