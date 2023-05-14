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

sr = 16000

n_fft = 2048
hop_length = 512

def preprocess_audio_file(audio_path):
    audio, _ = librosa.load(audio_path, sr=sr)

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    mag_spec = np.abs(stft)

    log_mag_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)

    return log_mag_spec

def preprocess_list_file(list_file_path, output_dir):
    with open(list_file_path, 'r') as f:
        audio_paths = f.read().splitlines()

    for audio_path in audio_paths:
        command = os.path.basename(os.path.dirname(audio_path))
        command_dir = os.path.join(output_dir, command)
        os.makedirs(command_dir, exist_ok=True)
        spectrogram = preprocess_audio_file(os.path.join(data_dir, audio_path))
        filename = os.path.splitext(os.path.basename(audio_path))[0] + nmp_ext
        np.save(os.path.join(command_dir, filename), spectrogram)

# Preprocess the training set
preprocess_list_file(train_list, os.path.join(spectogram_dir, train_dir))

# Preprocess the validation set
preprocess_list_file(validation_list, os.path.join(spectogram_dir, validation_dir))

# Preprocess the test set
preprocess_list_file(test_list, os.path.join(spectogram_dir, test_dir))