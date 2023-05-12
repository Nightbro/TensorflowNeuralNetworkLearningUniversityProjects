# install the required packages - if used from here, we will need to change data file url inside install_requirements
from requirements.install_requirements import install_requirements



import os
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import wget
import tarfile

import os
#import pathlib

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#import tensorflow as tf

#from tensorflow.keras import layers
#from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
#tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = 'data/mini_speech_commands'

mp3_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train"
segment_dir = "D:/Downloads/Edge/en/cv_corpus_v1/cv-valid-train-segments"
segment_length_ms = 3000



#run only once
def configureSetup(): 
    install_requirements()


def downloadDataSet():
    # Set the URL for the dataset download
    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'

    # Set the filename for the downloaded file
    filename = 'speech_commands_v0.02.tar.gz'
    output_path = os.path.join(os.getcwd(), 'speech_commands')

    # Download the file
    wget.download(url, filename)

    # Extract the files
    tar = tarfile.open(filename)
    tar.extractall(path=output_path)
    tar.close()





def main():
    print("Hello, World!")
    configureSetup();
    downloadDataSet()
    print("Kraj main funkcije")

if __name__ == "__main__":
    main()


