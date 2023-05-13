import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os



# Define the paths to the training, validation, and testing directories
data_dir = 'spectograms'
#train_dir = os.path.join(data_dir, 'train')
train_dir = os.path.join(data_dir, 'validation')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'testing')

# Define the batch size and number of epochs
batch_size = 32
epochs = 10



x_train = []
y_train = []

# Loop through all subdirectories
for subdir in os.listdir(train_dir):
    subpath = os.path.join(train_dir, subdir)
    # Loop through all files in the subdirectory
    for file in os.listdir(subpath):
        x_train.append(np.load(os.path.join(subpath,file)))
        y_train.append(subdir)

x_train = np.array(x_train)
y_train = np.array(y_train)



x_val = []
y_val = []

# Loop through all subdirectories
for subdir in os.listdir(val_dir):
    subpath = os.path.join(val_dir, subdir)
    # Loop through all files in the subdirectory
    for file in os.listdir(subpath):
        x_val.append(np.load(os.path.join(subpath,file)))
        y_val.append(subdir)


x_val = np.array(x_val)
y_val = np.array(y_val)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Train the model on the training set
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val))



x_test = []
y_test = []


# Loop through all subdirectories
for subdir in os.listdir(test_dir):
    subpath = os.path.join(test_dir, subdir)
    # Loop through all files in the subdirectory
    for file in os.listdir(subpath):
        x_test.append(np.load(os.path.join(subpath,file)))
        y_test.append(subdir)

x_test = np.array(x_test)
y_test = np.array(y_test)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_loss);
print(test_acc)
model.save('my_model.h5')
