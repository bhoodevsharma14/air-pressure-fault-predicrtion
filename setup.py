from setuptools import find_package,setup

from typing import List

REQUIREMENT_FILE_NAME="requirements.txt"


def get_requirements()->List[str]:
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.read_line()
        requirement_list = [requirement_name.replace("\n","") for requirement_list in requirement_list]

setup(
    name="sensor",
    version="0.0.1",
    author="bhoodev",
    author_email="bhoodev.sharma14@gmail.com",
    packages=find_package(),
    install_requires= get_requirements()
)