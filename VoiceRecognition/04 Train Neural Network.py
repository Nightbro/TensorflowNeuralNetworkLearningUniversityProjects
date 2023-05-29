import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os
from tensorflow.keras.optimizers import SGD

import numpy as np

max_frames = 128
n_fft = 256
hop_length = 128


# Load the preprocessed data

data = np.load('preprocessed_data_train.npz')
x_train, y_train = data['x_train'], data['y_train']

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_train = to_categorical(y_train, num_classes=35)
print(y_train.shape)

# Reshape the input tensor to have 4 dimensions
#print(x_train.shape)
x_train = x_train.reshape(-1, 129, 128, 1)



data = np.load('preprocessed_data_val.npz')
x_val, y_val = data['x_val'], data['y_val']

x_val = x_val.reshape(-1, 129, 128, 1)

y_val = label_encoder.transform(y_val)
y_val = to_categorical(y_val, num_classes=35)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(129, 128,1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(35, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9),
              metrics=['accuracy'])

# Train the model on the preprocessed data
batch_size = 32
epochs = 20
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
#                    validation_data=None)
                    validation_data=(x_val, y_val))

# Save the trained model
model.save('my_model_high.h5')
