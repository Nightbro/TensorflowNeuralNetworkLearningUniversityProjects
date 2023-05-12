#from requirements.install_requirements import install_requirements
# install the required packages - if used from here, we will need to change data file url inside install_requirements
#install_requirements()
import os
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


mp3_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train"
segment_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train-segments"
segment_length_ms = 3000



def audio_to_spectrogram(audio_path, output_folder, output_file_name):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute spectrogram
    # n_fft: The length of the FFT window
    # hop_length: The number of samples between successive frames
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    S = np.abs(S)

    # normalize the spectrogram
    S = librosa.util.normalize(S)

    spec_file = os.path.join(output_folder, output_file_name)
    np.save(spec_file, S)
    print(output_file_name)


def split_audio_files_mp3(audio_dir, segment_dir, segment_length_ms):
    # create the output directory if it doesn't exist
    os.makedirs(segment_dir, exist_ok=True)

    # loop over all audio files in the directory
    for audio_file in os.listdir(audio_dir):
        # load the audio file
        audio_path = os.path.join(audio_dir, audio_file)
        audio = AudioSegment.from_file(audio_path, format="mp3")

        # calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        # split the audio file into segments
        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = start_time + segment_length_ms
            segment = audio[start_time:end_time]

            # convert audio segment to spectrogram and save as image file
            audio_to_spectrogram(audio_path, segment_dir, f"{audio_file[:-4]}_{i}.npy")

def main():
    print("Hello, World!")
    split_audio_files_mp3(mp3_dir, segment_dir, segment_length_ms)
    print("Kraj main funkcije")

if __name__ == "__main__":
    main()


