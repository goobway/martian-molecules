import numpy as np
import pandas as pd

DATA_PATH = 'nasa-data/'

def convert_to_2d(df, output_dir):
    sample_ids = df.index.unique()
    for sample_id in sample_ids:
        # Get the row corresponding to the sample_id
        row = df.loc[sample_id]
        
        # Reshape the row into a 2D array
        image_2d = np.reshape(row.values, (350, 50))
        
        # Scale values from 0-1 to 0-255 and convert to uint8
        image_2d = (image_2d * 255).astype(np.uint8)
        
        # Stack to 3 dims to mimic rgb image
        image_2d_rgb = np.stack((image_2d,) * 3, axis=-1)
        
        # Save the 3D array as a .npy file
        filepath = output_dir + str(sample_id) + '.npy'
        np.save(filepath, image_2d_rgb)

# Load the preprocessed data
df = pd.read_csv(DATA_PATH + 'preprocessed_train_features.csv', skiprows=2, index_col='sample_id')

# Convert the data into 2D images
convert_to_2d(df, DATA_PATH + '2d_images/')