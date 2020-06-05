.. vhh_cmc documentation master file, created by
   sphinx-quickstart on Wed May  6 18:41:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Package Overview
================

The following description gives an overview of the folder structure of this python repository:

*name of repository*: vhh_cmc

   * **ApiSphinxDocumentation/**: includes all files to generate the documentation as well as the created documentations (html, pdf)
   * **config/**: this folder includes the required configuration file
   * **cmc/**: this folder represents the shot-type-classification module and builds the main part of this repository
   * **Demo/**: this folder includes a demo script to demonstrate how the package have to be used in customized applications
   * **Develop/**: includes scripts to generate the sphinx documentation. Furthermore, a script is included to run a
                   process to evaluate the implemented approach on a specified dataset.
   * **README.md**: this file gives a brief description of this repository (e.g. link to this documentation)
   * **requirements.txt**: this file holds all python lib dependencies and is needed to install the package in your own virtual environment
   * **setup.py**: this script is needed to install the cmc package in your own virtual environment

Setup  instructions
===================

This package includes a setup.py script and a requirements.txt file which are needed to install this package for custom
applications. The following instructions have to be done to use this library in your own application:

**Requirements:**

   * Ubuntu 18.04 LTS
   * python version 3.6.x

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_cmc/)
   * python3 -m venv /xxx/vhh_cmc/

**Activate the environment:**

   * source /xxx/vhh_cmc/bin/activate

**Checkout vhh_cmc repository to a specified folder:**

   * git clone https://github.com/dahe-cvl/vhh_cmc

**Install the cmc package and all dependencies:**

   * change to the root directory of the repository (includes setup.py)
   * python setup.py install

**Setup environment variables:**

   * source /data/dhelm/python_virtenv/vhh_sbd_env/bin/activate
   * export CUDA_VISIBLE_DEVICES=1
   * export PYTHONPATH=$PYTHONPATH:/XXX/vhh_cmc/:/XXX/vhh_cmc/Develop/:/XXX/vhh_cmc/Demo/

.. note::

  You can check the success of the installation by using the commend *pip list*. This command should give you a list
  with all installed python packages and it should include *vhh_cmc*.

**Run demo script**

   * change to root directory of the repository
   * python Demo/vhh_cmc_run_on_single_video.py


API Description
===============

This section gives an overview of all classes and modules in *cmc* as well as an code description.

.. toctree::
   :maxdepth: 4

   Configuration.rst
   CMC.rst
   OpticalFlow.rst
   PreProcessing.rst
   Evaluation.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**********
References
**********
.. target-notes::

.. _`Visual History of the Holocaust`: https://www.vhh-project.eu/
.. _`vhh_core`: https://github.com/dahe-cvl/vhh_core
.. _`vhh_stc`: https://github.com/dahe-cvl/vhh_stc

