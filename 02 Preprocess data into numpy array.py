import os
import librosa
import numpy as np

data_dir = 'speech_commands'
train_list = os.path.join(data_dir, 'train_list.txt')
validation_list = os.path.join(data_dir, 'validation_list.txt')
test_list = os.path.join(data_dir, 'testing_list.txt')

# Set the sampling rate for the audio files
sr = 16000

# Set the window size and hop length for the STFT
n_fft = 256
hop_length = 128

# Set the maximum number of frames for all spectrograms
max_frames = 128

# Define a function to preprocess a single audio file
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

    # Apply any necessary preprocessing to the log-magnitude spectrogram
    # (e.g., normalization, feature scaling, data augmentation)

    return log_mag_spec

# Define a function to preprocess all audio files in a list file
def preprocess_list_file(list_file_path):
    x_data = []
    y_data = []

    # Open the list file and read the paths to the audio files
    with open(list_file_path, 'r') as f:
        audio_paths = f.read().splitlines()

    # Loop over each audio file path in the list
    for audio_path in audio_paths:
        # Get the command label from the path
        command = os.path.basename(os.path.dirname(audio_path))

        # Preprocess the audio file and append to the x_data array
        spectrogram = preprocess_audio_file(os.path.join(data_dir, audio_path))
        x_data.append(spectrogram)
        y_data.append(command)

    # Stack all spectrograms to create a 4D tensor
    x_data = np.stack(x_data, axis=0)

    return x_data, np.array(y_data)

# Call the preprocessing function on your training, validation, and testing directories

x_val, y_val = preprocess_list_file(validation_list)
np.savez_compressed('preprocessed_data_val.npz', x_val=x_val, y_val=y_val)

x_test, y_test = preprocess_list_file(test_list)
np.savez_compressed('preprocessed_data_test.npz', x_test=x_test, y_test=y_test)


x_train, y_train = preprocess_list_file(train_list)
np.savez_compressed('preprocessed_data_train.npz', x_train=x_train, y_train=y_train)



# Save the preprocessed data as one compressed NumPy file
#np.savez_compressed('preprocessed_data_test.npz', x=x_test, y=y_test)
#np.savez_compressed('preprocessed_data.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
