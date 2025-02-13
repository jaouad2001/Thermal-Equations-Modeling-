
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
Lx = Ly = 10  # Plate dimensions (m)
Nx = 150
Ny = 150    # Number of grid points
dx = Lx / (Nx - 1)  # Grid spacing
dy = Ly / (Ny - 1)
alpha = 1e-4   # Thermal diffusivity (m^2/s)
dt = 0.25 * dx**2 / alpha  # Stable time step (CFL condition)
Nt = 500       # Number of time steps

# Initialize temperature field
T = np.zeros((Nx, Ny))

# Boundary conditions (Dirichlet)
T[:, 0] = 100  # Left boundary
T[:, -1] = -150  # Right boundary
T[0, :] = 159    # Bottom boundary
T[-1, :] = 0   # Top boundary

# Function to update temperature field using FDM (Explicit Scheme)
def update_temperature(T, alpha, dx, dy, dt, Nx, Ny):
    T_new = T.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )
    return T_new

# Run the simulation
T_history = [T.copy()]
for _ in range(Nt):
    T = update_temperature(T, alpha, dx, dy, dt, Nx, Ny)
    T_history.append(T.copy())

# Extract temperature profiles at specific locations
Tx = T_history[-1][Nx//2, :]
Ty = T_history[-1][:, Ny//2]

# Plot temperature profile along vertical centerline (Tx)
fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0, Ly, Ny), Tx, label="Ty (Centerline Temperature along Y)", color='r')
ax1.set_xlabel("Y Position (m)")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Temperature Profile along Vertical Centerline")
ax1.legend()
ax1.grid(True)

# Plot temperature profile along horizontal centerline (Ty)
fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(0, Lx, Nx), Ty, label="Tx (Centerline Temperature along X)", color='b')
ax2.set_xlabel("X Position (m)")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title("Temperature Profile along Horizontal Centerline")
ax2.legend()
ax2.grid(True)

plt.show()
