try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession
import logging
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

command = "";
try:
    install_reqs = parse_requirements("requirements.txt", session=PipSession())
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

# prepare pip command to install dependencies
tmp_str = ""
for requirement in install_reqs:
    tmp_str = tmp_str + " " + str(requirement.req)
command = "pip install " + tmp_str + " "

# execute dependendcies
os.system(command)
print("all dependencies installed!")

# install sbd package
setup(
     name='vhh_cmc',
     version='1.0.0',
     author="Daniel Helm",
     author_email="daniel.helm@tuwien.ac.at",
     description="Camera Movements Classification Package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dahe-cvl/vhh_cmc",
     packages=["cmc"]
)
