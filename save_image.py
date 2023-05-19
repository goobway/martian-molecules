import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = 'nasa-data/'
SAVE_PATH = 'nasa-data/2d_images_png/'  # specify the path to the directory where you want to save images
image_name = 'S0597.png'

# Create the directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

# Load the numpy array from the .npy file
image_2d = np.load(DATA_PATH + '2d_images/S0597.npy')

# Display the array as an image
plt.imshow(image_2d, aspect='auto')

# Add labels and title
plt.xlabel('Time Bin')
plt.ylabel('Rounded Mass')
plt.title('S0597 2D Mass Spectrum')

# Save the image
plt.savefig(SAVE_PATH + image_name, dpi=300)

plt.show()
