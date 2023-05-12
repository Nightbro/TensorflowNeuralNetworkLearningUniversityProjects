import pip

# read the requirements.txt file and install the required packages
def install_requirements():
    print("Requirements are being installed")
    with open("requirements/requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    pip.main(['install'] + requirements)
    print("Requirements have been installed")


#install_requirements()