import matplotlib.pyplot as plt
import numpy as np
import roar_py_interface
from typing import List
import os
from progress.bar import IncrementalBar
import transforms3d as tr3d

def estimate_curvature(points: np.ndarray) -> np.ndarray:
    """Return curvature estimate for each point using angle between segments."""
    curv = np.zeros(len(points))
    # avoid endpoints
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        # normalize
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            v1n = v1 / np.linalg.norm(v1)
            v2n = v2 / np.linalg.norm(v2)
            # angle between
            dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            angle = np.arccos(dot)
            curv[i] = angle
    return curv

print("\nLoading Waypoints\n")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
waypoints = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(os.path.join(SCRIPT_DIR, "waypoints", "waypointsPrimary.npz"))
)
track = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load(os.path.join(SCRIPT_DIR, "waypoints", "Monza Original Waypoints.npz"))
)

# Convert track to Nx2 array for curvature computation
track_xy = np.array([[wp.location[0], wp.location[1]] for wp in track])
curvature = estimate_curvature(track_xy)

# Normalize curvature for color mapping
curv_norm = curvature / (curvature.max() + 1e-9)

totalPoints = len(track)
progressBar = IncrementalBar("Plotting points", max=totalPoints)

plt.figure(figsize=(11, 11))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()

# Plot colored curvature line - use thicker lines and plot ONLY the track
for i in range(len(track) - 1):
    c = plt.cm.plasma(curv_norm[i])   # color proportional to turn sharpness
    seg = track_xy[i:i+2]
    plt.plot(seg[:, 0], seg[:, 1], color=c, linewidth=5)
    progressBar.next()

progressBar.finish()

# Optionally plot waypoints as small markers (uncomment if you want to see them)
# Plotting them AFTER the track so they appear on top
SHOW_PRIMARY_WAYPOINTS = False  # Set to True to show your custom waypoints
SHOW_HEADINGS = False           # Set to True to show direction arrows

if SHOW_PRIMARY_WAYPOINTS:
    print("Plotting primary waypoints...")
    for i in waypoints:
        plt.plot(i.location[0], i.location[1], "ko", markersize=2, alpha=0.5)

if SHOW_HEADINGS:
    print("Plotting headings...")
    # Plot headings (every Nth waypoint to avoid clutter)
    for idx, waypoint in enumerate(track):
        if idx % 20 == 0:  # only every 20th waypoint
            waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array([1, 0, 0])
            plt.arrow(
                waypoint.location[0],
                waypoint.location[1],
                waypoint_heading[0] * 10,
                waypoint_heading[1] * 10,
                width=2,
                color="black",
                alpha=0.6,
            )

print()

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="plasma")
sm.set_array(curvature)
plt.colorbar(sm, label="Turn Sharpness (Curvature)")

plt.title("Monza Track - Curvature Heatmap")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
