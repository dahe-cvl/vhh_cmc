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

Experiment 1:
Most Common Angle + Random Features + LK Optical Flow (pescoller)

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.91    |   0.51  |    0.65   |   182    |
| tilt |    0.50    |   0.79  |    0.61   |    78    |


|     accuracy   |      |      | 0.60   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.47 | 0.44 | 0.42   |    260 |
| weighted avg   | 0.79 | 0.60 | 0.64   |    260 |


Experiment 2:
Most Common Angle + GoodFeatures(Shi Tomasi Corner) + LK Optical Flow (pescoller)

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.93    |   0.64  |    0.76   |   182    |
| tilt |    0.69    |   0.77  |    0.73   |    78    |


|     accuracy   |      |      | 0.68   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.54 | 0.47 | 0.50   |    260 |
| weighted avg   | 0.86 | 0.68 | 0.75   |    260 |



Experiment 3: 
ORB Features + BFmatcher 

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    1.00    |   0.75  |    0.86   |   182    |
| tilt |    0.73    |   0.99  |    0.84   |    78    |


|     accuracy   |      |      | 0.82   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.58 | 0.58 | 0.56   |    260 |
| weighted avg   | 0.92 | 0.82 | 0.85   |    260 |


Experiment 4:
SIFT Features + knnMatcher

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    1.00    |   0.75  |    0.86   |   182    |
| tilt |    0.79    |   1.00  |    0.88   |    78    |


|     accuracy   |      |      | 0.82   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.60 | 0.59 | 0.58   |    260 |
| weighted avg   | 0.94 | 0.83 | 0.87   |    260 |


Experiment 5:
SURF features + knn matcher

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    1.00    |   0.74  |    0.85   |   182    |
| tilt |    0.80    |   1.00  |    0.89   |    78    |


|     accuracy   |      |      | 0.82   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.60 | 0.58 | 0.58   |    260 |
| weighted avg   | 0.94 | 0.82 | 0.86   |    260 |

Experiment 5:
BRIEF features + knn matcher

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.97    |   0.73  |    0.83   |   182    |
| tilt |    0.75    |   0.94  |    0.83   |    78    |


|     accuracy   |      |      | 0.82   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.57 | 0.55 | 0.55   |    260 |
| weighted avg   | 0.91 | 0.79 | 0.83   |    260 |


### Experiments 10.7.2020

all experiments with sift features + magnitude optimization

#################
TH = 5

NOT DONE YET

#################
TH = 3

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.99    |   0.85  |    0.91   |   182    |
| tilt |    0.98    |   0.69  |    0.81   |    78    |


|     accuracy   |      |      | 0.80   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.66 | 0.51 | 0.58   |    260 |
| weighted avg   | 0.99 | 0.80 | 0.88   |    260 |


#################
TH = 2

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.98    |   0.97  |    0.98   |   182    |
| tilt |    0.99    |   0.86  |    0.92   |    78    |


|     accuracy   |      |      | 0.93   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.66 | 0.61 | 0.63   |    260 |
| weighted avg   | 0.98 | 0.93 | 0.96   |    260 |



#################
TH = 1

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.98    |   0.99  |    0.99   |   182    |
| tilt |    0.99    |   0.90  |    0.94   |    78    |


|     accuracy   |      |      | 0.96   |    260 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.66 | 0.63 | 0.64   |    260 |
| weighted avg   | 0.98 | 0.96 | 0.97   |    260 |


**complete dataset 1213 samples (pan and tilt)**

ORB features and TH=2

|      | precision  | recall  | f1-score  | support  |
|------|------------|---------|-----------|----------|
| na   |    0.00    |   0.00  |    0.00   |     0    |
| pan  |    0.99    |   0.94  |    0.96   |   859    |
| tilt |    0.87    |   0.96  |    0.91   |   354    |


|     accuracy   |      |      | 0.94   |    1213 |
|----------------|------|------|--------|--------|
|    macro avg   | 0.62 | 0.63 | 0.63   |    1213 |
| weighted avg   | 0.95 | 0.94 | 0.95   |    1213 |