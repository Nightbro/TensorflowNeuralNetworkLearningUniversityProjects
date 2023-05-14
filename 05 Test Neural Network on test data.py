from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np

model = load_model('my_model_high.h5')

data = np.load('preprocessed_data_train.npz')
x_train, y_train = data['x_train'], data['y_train']
data_test = np.load('preprocessed_data_test.npz')
x_test, y_test = data_test['x_test'], data_test['y_test']

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_test = label_encoder.transform(y_test)
y_test = to_categorical(y_test, num_classes=35)


test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_loss)
print(test_acc)