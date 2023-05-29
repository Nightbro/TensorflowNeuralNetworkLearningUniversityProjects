import pandas as pd
import numpy as np
import tensorflow as tf

train_csv_file_path = "emnist-balanced-train.csv"
test_csv_file_path = "emnist-balanced-test.csv"

train_df = pd.read_csv(train_csv_file_path)
test_df = pd.read_csv(test_csv_file_path)

# Extract image data (pixel values) and label data for training set
X_train = train_df.iloc[:, 1:].values.astype('float32')
y_train = train_df.iloc[:, 0].values

# Normalize pixel values to a range between 0 and 1
X_train /= 255.0

# Reshape the image data into 2D arrays (28x28 pixels)
X_train = X_train.reshape(-1, 28, 28, 1)

X_test = test_df.iloc[:, 1:].values.astype('float32')
y_test = test_df.iloc[:, 0].values
X_test /= 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Save the preprocessed data for later use
np.savez("preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
