import numpy as np
import os

# Path to the .npz file
path = "/run/user/1000/gvfs/smb-share:server=truenas.local,share=arrakis/ece579/ece579_data_norm/norm_trajectory_0000999.npz"

# Check if file exists
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    # Load the .npz file
    data = np.load(path)

    # Print all array names
    print("Available arrays:", data.files)

    # Print the shape and contents of each array
    for key in data.files:
        array = data[key]
        print(f"\nArray '{key}': shape = {array.shape}, dtype = {array.dtype}")
        print(array)

    # If you want to explicitly print just the position size:
    if 'position' in data.files:
        print(f"\n'Size of position array: {data['position'].shape}'")