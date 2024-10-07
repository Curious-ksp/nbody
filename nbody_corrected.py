### Our Workplace
# Hi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### Constants
G = 6.6743e-11
AU = 1.496e11
day = 24*3600

### Font
universal = 14

### Initial Conditions
# Object 1
x1, y1, z1 = [0,0,0]
vx1, vy1, vz1 = [0,0,0]
# Object 2
x2, y2, z2 = [1*AU,0,0]
vx2, vy2, vz2 = [0,30000,0]


### Celestial bodies
bodies = [
    {"name": "Sun", "mass": 1.99e30, "position": [x1,y1,z1], "velocity": [vx1,vy1,vz1]},
    {"name": "Earth", "mass": 5.97e24, "position": [x2,y2,z2], "velocity": [vx2, vy2, vz2]}
]

### Extract Mass, Position & Velocities (into usable arrays)
masses = np.array([body["mass"] for body in bodies])
positions = np.array([body["position"] for body in bodies])
velocities = np.array([body["velocity"] for body in bodies])

### Time
dt = 0.1*day # time steps in seconds
t = 0
run_time = 1000*day
time_array = np.arange(0, run_time+dt, dt)

past_positions = [positions.copy()]
past_velocities = [velocities.copy()]
time_values = [t]

### HOW DOES GRAVITY WORK?!!
# Center of Mass
def compute_centerofmass(positions, masses):
    total_mass = np.sum(masses)
    center_of_mass = np.sum(positions.T*masses, axis=1)/total_mass
    return center_of_mass

# The Forces!
def eq_motion(positions, velocities, masses):
    dv_dt = np.zeros_like(positions)
    for i in range(len(masses)):
        for j in range(len(masses)):
            if i != j:
                rij = positions[j] - positions[i]
                r_mag = np.linalg.norm(rij)
                dv_dt[i] += (G * masses[j] / (r_mag**3)) * rij
    return dv_dt

### Equation of Motion Solver!
def solver(positions, velocities, masses, dt):
    # Compute accelerations (from equations of motion)
    acc = eq_motion(positions, velocities, masses)
    
    # Update positions and velocities
    new_positions = positions + velocities * dt
    new_velocities = velocities + acc * dt

    return new_positions, new_velocities

### Simulation/Calculation Time!
while t <= run_time:
    # Calculate new positions
    positions, velocities = solver(positions, velocities, masses, dt)
    
    # Adjust positions to keep center of mass at [0,0,0]
    center_of_mass = compute_centerofmass(positions, masses)
    positions -= center_of_mass
    
    # Store the newly created pos & vel into array
    past_positions.append(positions.copy())
    past_velocities.append(velocities.copy())
    
    # Update time
    t += dt
    time_values.append(t)
    
### Convert positions into easy to use NumPy array
past_positions = np.array(past_positions)
# Boundary of Plot frame
border = 2*AU


# Plot
colors = ['orange','blue']  
fig, ax = plt.subplots(dpi=150, figsize=(6, 6))  # Adjust DPI and figsize for performance

lines = []
scatter_plots = []

def update(frame):
    for i, body in enumerate(bodies):
        trail = past_positions[:frame, i, :]
        lines[i].set_data(trail[:, 0], trail[:, 1])
        scatter_plots[i].set_data([past_positions[frame, i, 0]], [past_positions[frame, i, 1]])
    
    ax.set_title('Simulation of Orbits', fontsize=universal)
    ax.set_xlim(-border, border)
    ax.set_ylim(-border, border)
    ax.set_xlabel('X (m)', fontsize=universal)
    ax.set_ylabel('Y (m)', fontsize=universal)
    ax.grid(alpha=0.5)
    ax.set_aspect('equal')
    return lines + scatter_plots

def init():
    for line in lines:
        line.set_data([], [])
    for scatter in scatter_plots:
        scatter.set_data([], [])
    return lines + scatter_plots

# Create lines and scatter plots for each body with labels
for i, body in enumerate(bodies):
    line, = ax.plot([], [], alpha=0.5, color=colors[i])
    scatter, = ax.plot([], [], 'o', color=colors[i], label=body["name"])
    lines.append(line)
    scatter_plots.append(scatter)
    
# Add legend outside the plot
ax.legend(loc='upper left', fontsize=8)

# Use reduced frames to speed up the processc
ani = animation.FuncAnimation(fig, update, frames=range(0, len(time_values), 200), init_func=init, interval=10, repeat=False)





