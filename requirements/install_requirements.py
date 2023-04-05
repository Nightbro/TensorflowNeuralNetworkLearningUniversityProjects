import pip

# read the requirements.txt file and install the required packages
def install_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    pip.main(['install'] + requirements)


install_requirements()