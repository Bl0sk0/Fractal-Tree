#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:35:50 2024

@author: Pablo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Load the data from the 'tree.npy' file
X = np.load('tree.npy')

# Initial parameters
n = 2  # Number of iterations (depth of the tree)
r = 5  # Branching factor
m = (r**(n+1)-1) // (r-1)  # Total number of nodes in the tree
x = X[:,0].reshape(2,1,4)  # Reshape the first element of X for later use

# Configure the figure for the animation
fig, ax = plt.subplots(figsize=(9, 16))  # Aspect ratio 9:16 for social media format
fig.set_dpi(100)  # Set resolution
fig.set_size_inches(1080/fig.get_dpi(), 1920/fig.get_dpi())  # Set size for Instagram reels (1080x1920)

# Set the plot limits based on the data
xmin, xmax = np.min(X[0]), np.max(X[0])
ymin, ymax = np.min(X[1]), np.max(X[1])
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.axis('off')  # Hide axis

# Parameters for the zoom effect in the animation
tu = 2.5  # Time units for a branch to grow
zot = 2  # Zoom-out time in 'tu'
gvt = 1  # Time for the general view in 'tu'
tp = (n+1)/(n+1+zot)  # Transition point for zoom timing
enable = True  # Flag to enable zoom
shape = 1  # Zoom shape factor (linear zoom)
tmax = 1.3  # Maximum time for each iteration
fps = 24  # Frames per second for the animation

# Zoom function to handle the zoom effect during animation
def zoom_function(i, t):
    if enable:
        s = (i + t/tmax) / (n+1+zot)
        if s < tp:
            s /= tp
        else:
            s = (1-s)/(1-tp) if s <= 1 else 0
        s **= shape  # Adjust zoom factor by the shape
        
        # Apply zoom to x and y limits
        ax.set_xlim((1-s)*xmin + s*(-4.8), (1-s)*xmax + s*(-2.3))
        ax.set_ylim((1-s)*ymin + s*(3.9), (1-s)*ymax + s*(3.9 + (ymax - ymin) * 2.5 / (xmax - xmin)))

# Function to generate gradient colors between brown (branches) and pink (flowers)
def generate_colors(n):
    brown = np.array([0.104, 0.059, 0.016])  # Brown (for branches)
    pink = np.array([1, 183/255, 197/255])  # Pink (for cherry blossoms)
    t = np.power(np.linspace(0, 1, n+1), 3)  # Exponent to smooth color transition
    colors = np.outer(1 - t, brown) + np.outer(t, pink)  # Blend brown to pink
    return colors

# Generate colors for each iteration level
colors = generate_colors(n)

# List to store completed polygons (fully grown branches)
completed_polygons = []

# Function to update each frame of the animation
def update(frame):
    i, t = frame

    if i > n:
        # After all branches have grown, only zoom effect remains
        zoom_function(i, t)
        return
    
    # Calculate range of nodes for current iteration
    a = (r**i - 1) // (r - 1)
    b = (r**(i+1) - 1) // (r - 1)
    x_i = X[:, a:b, :]  # Select data for current iteration
    
    # Interpolate between points to animate growth
    z = x_i.copy()
    z[:,:,2] = x_i[:,:,2]*t + (1-t)*x_i[:,:,1]
    z[:,:,3] = x_i[:,:,3]*t + (1-t)*x_i[:,:,0]

    # Remove any polygons that are not completed yet
    for patch in ax.patches[:]:
        if patch not in completed_polygons:
            patch.remove()
    
    # Add new polygons for current frame
    for j in range(r**i):
        polygon = plt.Polygon(z[:,j].T, closed=True, fill=True, edgecolor=None, facecolor=colors[i])
        ax.add_patch(polygon)
        
        # Mark polygon as completed if t reaches tmax (fully grown)
        if t == tmax:
            completed_polygons.append(polygon)
    
    # Apply zoom effect for this frame
    zoom_function(i, t)

# Create frames for the animation, each frame is a tuple (i, t)
frames = [(i, t) for i in range(n+1+zot+gvt) for t in np.linspace(0, tmax, num=round(tu*fps))]

# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=frames, repeat=True)

# Save the animation to an MP4 file
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Pablo'), bitrate=1800)
ani.save(f'./animations/tree_animation_zoom_n{n}.mp4', writer=writer)
