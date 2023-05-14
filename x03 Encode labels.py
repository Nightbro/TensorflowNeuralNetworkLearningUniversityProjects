from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import os
from keras.utils import to_categorical

label_encoder = LabelEncoder()

data = np.load('preprocessed_data_val.npz')
x_val, y_val = data['x_val'], data['y_val']

label_encoder.fit(y_val)
y_val = label_encoder.transform(y_val)
print("after labl")
print(y_val)
y_val = to_categorical(y_val, num_classes=35)

print("after cat")
print(y_val)
