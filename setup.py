from setuptools import find_packages,setup
from typing import List

def get_requirements(filepath:str) -> List[str]:

    requirements = []

    HYPEN_E_DOT = "-e ."
    
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(

    name="mlproject",
    version='0.0.1',
    author="Lakshay",
    author_email="iamlakshaygoel5990@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)