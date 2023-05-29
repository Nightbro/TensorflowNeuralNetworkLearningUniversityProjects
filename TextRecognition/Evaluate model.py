import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the preprocessed data
data = np.load("preprocessed_data.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# Load the trained model
model = tf.keras.models.load_model("handwriting_model.h5")


y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_labels)
print("Test Accuracy:", accuracy)

# Decode the labels using the mapping file
label_mapping = {}
with open("emnist-balanced-mapping.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        label, char = line.strip().split()
        label_mapping[int(label)] = chr(int(char))

y_test_decoded = np.array([label_mapping[label] for label in y_test])
y_pred_decoded = np.array([label_mapping[label] for label in y_pred_labels])

print("True Labels:", y_test_decoded[:10])
print("Predicted Labels:", y_pred_decoded[:10])