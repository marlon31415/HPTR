import h5py
import matplotlib.pyplot as plt
import numpy as np


# Function to read and plot polylines from h5 file
def plot_polylines_from_h5(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        # Access the dataset containing polylines
        scene_tokens = list(h5_file.keys())
        polylines = h5_file[f"{scene_tokens[1000]}/route/pos"]

        # Plot each polyline
        plt.figure(figsize=(10, 8))
        for polyline in polylines:
            # Filter out [0, 0] padding
            polyline = np.array([point for point in polyline if not np.all(point == 0)])

            # If there are valid points remaining, plot the polyline
            if len(polyline) > 0:
                x, y = zip(*polyline)  # Unpack x and y coordinates
                plt.plot(x, y, marker="o", markersize=2)  # Plot the polyline

        # Set plot labels and title
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Polylines from /route/pos")
        plt.grid(True)
        plt.axis("equal")  # Maintain aspect ratio

        # Show the plot
        plt.savefig("./centerlines.pdf")
        plt.close()


# Specify the path to your h5 file
file_path = "/mrtstorage/datasets_tmp/nuplan_hptr/training_mini.h5"

# Call the function to plot polylines
plot_polylines_from_h5(file_path)
