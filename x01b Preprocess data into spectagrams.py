import os
import librosa
import numpy as np

data_dir = 'speech_commands'
spectogram_dir='spectograms'
train_dir='train'
test_dir='test'
validation_dir='validation'
nmp_ext='.npy'
train_list = os.path.join(data_dir, 'train_list.txt')
validation_list = os.path.join(data_dir, 'validation_list.txt')
test_list = os.path.join(data_dir, 'testing_list.txt')

# Set the sampling rate for the audio files
sr = 16000

# Set the window size and hop length for the STFT
n_fft = 2048
hop_length = 512

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

    # Apply any necessary preprocessing to the log-magnitude spectrogram
    # (e.g., normalization, feature scaling, data augmentation)

    return log_mag_spec

# Define a function to preprocess all audio files in a list file
def preprocess_list_file(list_file_path, output_dir):
    # Open the list file and read the paths to the audio files
    with open(list_file_path, 'r') as f:
        audio_paths = f.read().splitlines()

    # Loop over each audio file path in the list
    for audio_path in audio_paths:
        # Get the command label from the path
        command = os.path.basename(os.path.dirname(audio_path))

        # Create the output directory if it does not exist
        command_dir = os.path.join(output_dir, command)
        os.makedirs(command_dir, exist_ok=True)

        # Preprocess the audio file and save it as a NumPy file
        spectrogram = preprocess_audio_file(os.path.join(data_dir, audio_path))
        filename = os.path.splitext(os.path.basename(audio_path))[0] + nmp_ext
        np.save(os.path.join(command_dir, filename), spectrogram)

# Preprocess the training set
#preprocess_list_file(train_list, os.path.join(spectogram_dir, train_dir))

# Preprocess the validation set
preprocess_list_file(validation_list, os.path.join(spectogram_dir, validation_dir))

# Preprocess the test set
preprocess_list_file(test_list, os.path.join(spectogram_dir, test_dir))