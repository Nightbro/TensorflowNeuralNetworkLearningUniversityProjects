import os
import numpy as np

def preprocess_data(data_dir, type):
    x_data = []
    y_data = []

    for subdir in os.listdir(data_dir):
        subpath = os.path.join(data_dir, subdir)
        for file in os.listdir(subpath):
            spectrogram = np.load(os.path.join(subpath, file))
            # Apply any necessary preprocessing steps here
            x_data.append(spectrogram)
            y_data.append(subdir)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Save the preprocessed data as one or multiple files
    np.savez_compressed('preprocessed_data_'+type+'.npz', x=x_data, y=y_data)

# Call the preprocessing function on your training, validation, and testing directories
preprocess_data('spectograms/train', 'train')
preprocess_data('spectograms/validation', 'validation')
preprocess_data('spectograms/testing', 'testing')
