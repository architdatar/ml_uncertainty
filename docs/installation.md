Installation
============


<!-- Stable release
--------------

To install ML Uncertainty Quantification, run this command in your terminal:

.. code-block:: console

    $ pip install ml_uncertainty

This is the preferred method to install ML Uncertainty Quantification, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/ -->


Stable release
--------------

To install ML Uncertainty Quantification, run this command in your terminal:
```
pip install ml-uncertainty
```

This is the preferred method to install ML Uncertainty Quantification, as it will always install the most recent stable release.

From source
------------

#### Basic user installation
1. Clone / download this repo.
2. Install ml_uncertainty.
    
    Navigate to the ml_uncertainty folder from step 1 and in the terminal, type 
    ```
    pip install .
    ```
    The package should install automatically.

**CAUTION**: Please consider the dependencies and make sure that they don't conflict with your existing dependencies. 

**NOTE**: It is recommended that before doing step 2, you set up a virtual environment as shown below.

### Set up a new environment (Recommended)
Create and activate new virtual environment with the desired Python version. 


#### Using conda
##### Check current Python version
In the conda terminal, type:
```
python --version
```    

Ensure that it is one of the allowed versions. Else, use a different Python version.

##### Create environment
```
conda create -n <ENV_NAME> python=<REQUIRED PYTHON VERSION> 
```

##### Activate environment
```
conda activate <ENV_NAME>
```

#### Using Python
##### Check current Python version
In terminal, type:
```
python --version
```
Make sure that it is one of the allowed versions.

##### Create environment
```
python -m venv "PATH_TO_YOUR_ENV"
```

##### Activate environment
For Mac / Linux:
```
source "PATH_TO_YOUR_ENV/bin/activate"
```

For Windows:
```    
source "PATH_TO_YOUR_ENV/Scripts/Activate"
```

### Testing (OPTIONAL)
To test the package and ensure that it runs correctly, navigate to the package folder and type:
```
pytest
```

If it passes all tests, it means that the package has been correctly installed. 
