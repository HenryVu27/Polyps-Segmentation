import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# --- Configuration ---
DISK_RADIUS = 1.0
L_REMOVE = 2 # Number of disks to remove
R_ADD = 1  # Number of disks to add (R < L)

# --- Define Points to be Covered ---
# Let's place a few points that can initially be covered by 2 disks,
# but also by a single different disk.
points_to_cover = np.array([
    [0.1, 0.1],
    [0.5, 0.2],
    [0.2, 0.5]
])

# --- Define Candidate Unit Disks ---
# Centers of potential disks. We'll need at least:
# - two disks for the initial cover (let's call them D0, D1)
# - one different disk for the improved cover (D2)
candidate_disk_centers = np.array([
    # D0 (index 0): Covers point 0 and 2 initially
    [-0.1, 0.3],
    # D1 (index 1): Covers point 1 initially
    [0.7, 0.0],
    # D2 (index 2): Covers all points, will replace D0 and D1
    [0.3, 0.3]
])

# --- Define the Local Improvement Step ---
# Initial cover uses disks with indices 0 and 1 from candidate_disk_centers
initial_cover_indices = [0, 1]

# These are the *indices within the initial_cover_indices list*
# that correspond to disks we want to remove.
indices_within_initial_cover_to_remove = [0, 1] # Remove the 0th and 1st disk *in the initial_cover_indices list*

# This is the *index within the candidate_disk_centers array*
# that corresponds to the disk(s) we want to add.
indices_of_candidate_to_add = [2] # Add the disk with index 2 from candidate_disk_centers

# --- Verification Helper Function ---
def is_point_covered(point, disk_centers, radius):
    """Checks if a point is covered by any of the disks."""
    if len(disk_centers) == 0: return False # Handle empty cover case
    # Use broadcasting for efficient distance calculation
    distances_sq = np.sum((disk_centers - point)**2, axis=1)
    return np.any(distances_sq <= radius**2 + 1e-9) # Add tolerance for floating point

def check_cover(points, disk_centers, radius):
    """Checks if all points are covered by the given disks."""
    for point in points:
        if not is_point_covered(point, disk_centers, radius):
            return False
    return True

# --- Construct Initial and Final Covers ---
initial_cover_disk_centers = candidate_disk_centers[initial_cover_indices]

# Determine the actual indices from candidate_disk_centers that are removed
actual_removed_indices_from_candidates = [initial_cover_indices[i] for i in indices_within_initial_cover_to_remove]

# Calculate final cover indices
# Start with initial indices
initial_set_of_indices = set(initial_cover_indices)
removed_set_of_indices = set(actual_removed_indices_from_candidates)
added_set_of_indices = set(indices_of_candidate_to_add)

final_cover_indices = list((initial_set_of_indices - removed_set_of_indices) | added_set_of_indices)
final_cover_disk_centers = candidate_disk_centers[final_cover_indices]

# --- Verify the Scenario (Crucial!) ---
print(f"Initial cover has {len(initial_cover_indices)} disks (indices: {initial_cover_indices}).")
print(f"Disks removed (candidate indices): {actual_removed_indices_from_candidates}")
print(f"Disks added (candidate indices): {indices_of_candidate_to_add}")
print(f"Final cover has {len(final_cover_indices)} disks (indices: {final_cover_indices}).")

if len(initial_cover_indices) - len(final_cover_indices) != L_REMOVE - R_ADD:
    print(f"Error: The change in number of disks ({len(initial_cover_indices)} -> {len(final_cover_indices)}) does not match L-R ({L_REMOVE}-{R_ADD}={L_REMOVE-R_ADD}).")
    # exit() # Allow visualization even if numbers don't match

is_initial_valid = check_cover(points_to_cover, initial_cover_disk_centers, DISK_RADIUS)
is_final_valid = check_cover(points_to_cover, final_cover_disk_centers, DISK_RADIUS)

print(f"Initial cover is valid: {is_initial_valid}")
print(f"Final cover is valid: {is_final_valid}")

if not is_initial_valid or not is_final_valid:
     print("\nWarning: Either initial or final cover is not valid. Adjust points/disk centers or indices.")
     # The current setup is designed to work, so this warning shouldn't appear unless code is modified.

# Convert sets for highlighting checks
removed_indices_set = set(actual_removed_indices_from_candidates)
added_indices_set = set(indices_of_candidate_to_add)

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7)) # Make figure slightly wider

# Plot 1: Initial State
ax1 = axes[0]
ax1.set_title(f"Initial Cover ({len(initial_cover_indices)} disks)")
ax1.set_aspect('equal', adjustable='box')

# Plot points
ax1.scatter(points_to_cover[:, 0], points_to_cover[:, 1], color='black', s=50, zorder=5, label="Points to Cover")

# Plot initial cover disks
# Iterate through the indices present in the initial cover
for disk_idx in initial_cover_indices:
    center = candidate_disk_centers[disk_idx]
    is_removed = disk_idx in removed_indices_set

    color = 'skyblue'
    edgecolor = 'blue'
    linewidth = 1
    label = f"Disk {disk_idx}"
    if is_removed:
        edgecolor = 'red'
        linewidth = 2
        color = 'salmon'
        label += " (Removed)"

    circle = patches.Circle(center, DISK_RADIUS, facecolor=color, edgecolor=edgecolor,
                            linewidth=linewidth, alpha=0.5, label=label) # Increased alpha slightly
    ax1.add_patch(circle)

# Set limits and add labels
all_coords = np.vstack([points_to_cover, candidate_disk_centers])
min_x, min_y = np.min(all_coords, axis=0) - DISK_RADIUS - 0.5
max_x, max_y = np.max(all_coords, axis=0) + DISK_RADIUS + 0.5
# Ensure limits are not identical if only one point/disk exists
min_x = min(min_x, -2) # Default minimums if points are all near origin
min_y = min(min_y, -2)
max_x = max(max_x, 2) # Default maximums
max_y = max(max_y, 2)
ax1.set_xlim(min_x, max_x)
ax1.set_ylim(min_y, max_y)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
# Avoid duplicate labels in legend
# CORRECTED TYPO HERE: get_legend_handles_labels
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc='upper left') # Specify location
ax1.grid(True, linestyle='--', alpha=0.6)


# Plot 2: Final State after Local Improvement
ax2 = axes[1]
ax2.set_title(f"After Local Improvement ({len(final_cover_indices)} disks)")
ax2.set_aspect('equal', adjustable='box')

# Plot points
ax2.scatter(points_to_cover[:, 0], points_to_cover[:, 1], color='black', s=50, zorder=5) # No need to label again

# Plot final cover disks
# Iterate through the indices present in the final cover
for disk_idx in final_cover_indices:
    center = candidate_disk_centers[disk_idx]
    is_added = disk_idx in added_indices_set

    color = 'lightgreen'
    edgecolor = 'green'
    linewidth = 1
    label = f"Disk {disk_idx}"
    if is_added:
        edgecolor = 'darkgreen'
        linewidth = 2
        color = 'palegreen'
        label += " (Added)"

    circle = patches.Circle(center, DISK_RADIUS, facecolor=color, edgecolor=edgecolor,
                            linewidth=linewidth, alpha=0.5, label=label) # Increased alpha slightly
    ax2.add_patch(circle)

# Use the same limits as the first plot for consistent view
ax2.set_xlim(min_x, max_x)
ax2.set_ylim(min_y, max_y)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
# Avoid duplicate labels in legend
# CORRECTED TYPO HERE: get_legend_handles_labels
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper left') # Specify location
ax2.grid(True, linestyle='--', alpha=0.6)

# Add text description of the step
plt.suptitle(f"Local Search Step: Replace {L_REMOVE} disks with {R_ADD} disks (Reducing cover size)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

plt.show()