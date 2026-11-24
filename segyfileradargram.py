
import segyio
import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = r"C:\Users\carlo\Danmarks Tekniske Universitet\Autumn 2026 Project Family - Dokumenter\General\BEng_Malthe&Christian&Carlo\Data\GPR Data handling\GPR Handling\GPR_Data\Topscanning\Top Feardig_20261104_155441_L036_20261105_154221.sgy"

# Read SGY data
with segyio.open(file_path, "r", ignore_geometry=True) as f:
    data = f.trace.raw[:]

# Proceq GP8800 parameters
dt = 1 / 40.96       # Sampling interval [ns]
dx = 0.5             # Trace spacing [mm]

# Calibration parameters
epsilon_r = 7.8
t0 = 0.9731          # Time-zero correction [ns] # 0.9731 ns for første negative peak, 0.7910 ns ved første peak

# Electromagnetic wave velocity in concrete
c = 0.299792458      # Speed of light [m/ns]
v = c / np.sqrt(epsilon_r)

# Data dimensions
num_traces, num_samples = data.shape

# Original time axis [ns]
time_ns = np.arange(num_samples) * dt

# Corrected time axis [ns]
time_corrected = time_ns - t0

# Convert time to depth [mm]
depth_mm = (v * time_corrected / 2) * 1000

# Horizontal distance [mm]
distance_mm = np.arange(num_traces) * dx*10

# Amplitude clipping
clip = np.percentile(np.abs(data), 99)

# Plot radargram
fig, ax = plt.subplots(figsize=(12, 5))

im = ax.pcolormesh(
    distance_mm,
    depth_mm,
    data.T,
    shading="nearest",
    cmap="seismic",
    vmin=-clip,
    vmax=clip
)

# Show entire depth range, including negative depths
ax.set_ylim(depth_mm[-1], depth_mm[0])

# Mark concrete surface at zero depth
ax.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Concrete surface (0 mm)"
)

ax.set_xlabel("Distance along column [mm]")
ax.set_ylabel("Depth in concrete [mm]")
ax.set_title(
    f"GPR Radargram (εr = {epsilon_r}, t0 = {t0} ns)"
)

ax.legend(loc="lower right")
fig.colorbar(im, ax=ax, label="Amplitude")

plt.tight_layout()
plt.show()

# Calibration information
print(f"Dielectric constant: {epsilon_r}")
print(f"Time-zero correction: {t0} ns")
print(f"Wave velocity: {v:.5f} m/ns")
print(f"Minimum depth: {depth_mm[0]:.2f} mm")
print(f"Maximum depth: {depth_mm[-1]:.2f} mm")
