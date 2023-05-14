import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa

data = np.load('preprocessed_data_train.npz')
y_train = data['y_train']
model = tf.keras.models.load_model('my_model_high.h5')
max_frames = 128
n_fft = 256
hop_length = 128
sr = 16000


def preprocess_audio_file(audio_path):
    # Load the audio file as a NumPy array
    audio, _ = librosa.load(audio_path, sr=sr)

    # Apply the STFT to the audio signal
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Convert the complex-valued STFT output to a magnitude spectrogram
    mag_spec = np.abs(stft)

    # Convert the magnitude spectrogram to a log-magnitude spectrogram
    log_mag_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)

    # Trim or pad the log-magnitude spectrogram to ensure it has max_frames frames
    if log_mag_spec.shape[1] > max_frames:
        log_mag_spec = log_mag_spec[:, :max_frames]
    else:
        log_mag_spec = np.pad(log_mag_spec, ((0, 0), (0, max_frames - log_mag_spec.shape[1])), mode='constant')

    return log_mag_spec

audio_spectogram = preprocess_audio_file('test_7.wav')
#audio_spectogram = preprocess_audio_file('test_forward.wav')

# Reshape the input tensor to have 4 dimensions
audio_spectogram = audio_spectogram.reshape(-1, 129, 128, 1)

# Predict the label of the new audio sample
y_pred = model.predict(audio_spectogram)

# Retrieve the predicted label
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
predicted_label = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

print(predicted_label)