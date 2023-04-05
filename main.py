#from requirements.install_requirements import install_requirements
# install the required packages - if used from here, we will need to change data file url inside install_requirements
#install_requirements()
import os
from pydub import AudioSegment



mp3_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train"
segment_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train-segments"
segment_length_ms = 3000



def split_audio_files(audio_dir, segment_dir, segment_length_ms):
    os.makedirs(segment_dir, exist_ok=True)
    # loop over all audio files in the directory
    for audio_file in os.listdir(audio_dir):
        # load the audio file
        audio_path = os.path.join(audio_dir, audio_file)
        audio = AudioSegment.from_file(audio_path)

        # calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        # split the audio file into segments
        for i in range(num_segments):
            start_time = i * segment_length_ms
            end_time = start_time + segment_length_ms
            segment = audio[start_time:end_time]
            segment_path = os.path.join(segment_dir, f"{audio_file[:-4]}_{i}.wav")
            segment.export(segment_path, format="wav")


def main():
    print("Hello, World!")
    #split_audio_files(mp3_dir, segment_dir, segment_length_ms)
    print("Kraj main funkcije")

if __name__ == "__main__":
    main()


