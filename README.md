# Plugin package: Camera Movements Classification

This package includes all methods to classify a given shot/or image sequence in one of the categories Pan, Tilt or NA.

## Package Description

PDF format: [vhh_cmc_pdf](https://github.com/dahe-cvl/vhh_cmc/blob/master/ApiSphinxDocumentation/build/latex/vhhpluginpackagecameramovementsclassificationvhh_cmc.pdf)
    
HTML format (only usable if repository is available in local storage): [vhh_cmc_html](https://github.com/dahe-cvl/vhh_cmc/blob/master/ApiSphinxDocumentation/build/html/index.html)
    
    
## Quick Setup

**Requirements:**

   * Ubuntu 18.04 LTS
   * python version 3.6.x
   
### 0 Environment Setup (optional)

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_cmc/)
   * python3 -m venv /xxx/vhh_cmc/

**Activate the environment:**

   * source /xxx/vhh_cmc/bin/activate
   
### 1A Install using Pip

The VHH Shot Boundary Detection package is available on [PyPI](https://pypi.org/project/vhh-cmc/) and can be installed via ```pip```.

* Update pip and setuptools (tested using pip\==20.2.3 and setuptools==50.3.0)
* ```pip install vhh-cmc```

Alternatively, you can also build the package from source.

### 1B Install by building from Source

**Checkout vhh_cmc repository to a specified folder:**

   * git clone https://github.com/dahe-cvl/vhh_cmc

**Install the cmc package and all dependencies:**

   * Update ```pip``` and ```setuptools``` (tested using pip\==20.2.3 and setuptools==50.3.0)
   * Install the ```wheel``` package: ```pip install wheel```
   * change to the root directory of the repository (includes setup.py)
   * ```python setup.py bdist_wheel```
   * The aforementioned command should create a /dist directory containing a wheel. Install the package using ```python -m pip install dist/xxx.whl```
   
> **_NOTE:_**
You can check the success of the installation by using the commend *pip list*. This command should give you a list
with all installed python packages and it should include *vhh-cmc*.

### 2 Setup environment variables (optional)

   * source /data/dhelm/python_virtenv/vhh_sbd_env/bin/activate
   * export CUDA_VISIBLE_DEVICES=0
   * export PYTHONPATH=$PYTHONPATH:/XXX/vhh_cmc/:/XXX/vhh_cmc/Develop/:/XXX/vhh_cmc/Demo/

### 3 Run demo script (optional)

   * change to root directory of the repository
   * python Demo/vhh_cmc_run_on_single_video.py
   
## Release Generation

* Create and checkout release branch: (e.g. v1.1.0): ```git checkout -b v1.1.0```
* Update version number in setup.py
* Update Sphinx documentation and release version
* Make sure that ```pip``` and ```setuptools``` are up to date
* Install ```wheel``` and ```twine```
* Build Source Archive and Built Distribution using ```python setup.py sdist bdist_wheel```
* Upload package to PyPI using ```twine upload dist/*```

## Evaluation & Results

Experiment: "../cmc_eval_20210614/vhhmmsi_eval_db_part4/vhhmmsi_eval_db_2"

|  parameters    | values  | 
|----------------|------------|
| mvi_mv_ratio   |    0.2  |   
| threshold_significance  |    2.0  |  
| threshold_consistency  |    2.3  |  
| mvi_window_size  |    10  |  
| region_window_size  |    5  |  


|      | precision  | recall  | f1-score  | accuracy  |
|------|------------|---------|-----------|----------|
| exp02   |    0.7293    |   0.7244  |    0.7181   |     0.7093    |


