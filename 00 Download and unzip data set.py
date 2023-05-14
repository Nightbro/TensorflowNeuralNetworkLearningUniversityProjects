import wget
import tarfile
import os
import glob

def downloadDataSet():
    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    filename = 'speech_commands_v0.02.tar.gz'
    output_path = os.path.join(os.getcwd(), 'speech_commands')
    wget.download(url, filename)
    tar = tarfile.open(filename)
    tar.extractall(path=output_path)
    tar.close()

downloadDataSet()


def fixPath(path, newPath):
    with open(path, 'r') as f_in, open(newPath, 'w') as f_out:
        for line in f_in:
            # Replace backslashes with forward slashes in the file path
            line_corrected = line.replace('\\', '/')
            # Write the corrected file path to the output file
            f_out.write(line_corrected)

fixPath("speech_commands/testing_list.txt","speech_commands/testing.txt")
fixPath("speech_commands/validation_list.txt","speech_commands/validation.txt")


#after this remove files and folders from the txt file manualy, as well as data in _background_noise_
def createAFileWithAllFilePathsInIt():
    dir_path = 'speech_commands'
    file_path = 'speech_commands/train_list.txt'
    exclude_testing_path = 'speech_commands/testing_list.txt'
    exclude_validation_path = 'speech_commands/validation_list.txt'

    with open(exclude_testing_path, 'r') as exclude_file:
        exclude_testing_files = [line.strip() for line in exclude_file]
    with open(exclude_validation_path, 'r') as exclude_file:
        exclude_validation_files = [line.strip() for line in exclude_file]

    with open(file_path, 'w') as f:
        # Loop over each subdirectory in the specified directory
        for subdir_path in glob.glob(os.path.join(dir_path, '*'), recursive=True):
            # Check if the current path is a directory and not the root folder or the _background_noise_ folder
            if os.path.isdir(subdir_path) and os.path.basename(subdir_path) != '_background_noise_' and subdir_path != dir_path:
                # Loop over each file in the current subdirectory
                for file_path in glob.glob(os.path.join(subdir_path, '*')):
                    file_full_name=f"{os.path.basename(subdir_path)}/{os.path.basename(file_path)}"

                    # Check if the current file is not in the list of files to exclude
                    if file_full_name not in exclude_testing_files:
                        if file_full_name not in exclude_validation_files:
                            # Write the file path to the file
                            f.write(file_full_name+"\n")


createAFileWithAllFilePathsInIt()

