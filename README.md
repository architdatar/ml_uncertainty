
<!-- TODO : Make these dynamic-->
![Version badge](https://img.shields.io/badge/version-0.1.0-blue)
![Code style badge](https://img.shields.io/badge/code_style-black-black)
![Python bade](https://img.shields.io/badge/python-3.9-blue?logo=python)

ML Uncertainty
=============================
## ML model inference, error propagation, and non-linear modelling


ML Uncertainty is a Python module for machine learning inference build on top of scikit-learn and autograd packages, and is distributed under the MIT license. 

This package has been built by Archit Datar (archit.datar@celanese.com, architdatar@gmail.com). 

Intended audience
----
This package is intended to benefit data scientists and ML enthusiasts. 

Motivation
----
Too often in machine learning, we fit complex models, but cannot produce prediction intervals or feature significance. 

This is especially true of the scikit-learn environment which is extremely easy to use but does not offer these functionalities.  

However, in many use cases, especially where we have small and fat datasets, these are insights are critical to produce reliable models and insights. 

Enter ML Uncertainty! This provides an easy API to get all these insights from models. 

It takes scikit-learn fitted models as inputs and does the required statistics to generate these insights.


Features
--------

1. **Model parameter significance testing:** Tests whether the given model parameters are truly significant or not.

     For ensemble models, it can inform if given features are truly important or if they just seem so due to the instability of the model.

2. **Prediction intervals:** Can produce prediction and confidence intervals for model predictions.

3. **Error propagation:** For a model, what would be the prediction intervals look like given uncertainty in input data and / or model parameters? 

4. **Non-Linear regression:** Provided a scikit-learn-style API to fit non-linear models. These are often encountered in scientific applications. 

Installation
------------
### Dependencies

ml_uncertainty requires:

* Python (3.9)
* Numpy (1.20.3)
* scipy (1.7.1)
* pandas (1.5.3)
* scikit-learn (1.0.2)
* autograd (1.3)

### User installation

#### Basic user installation
1. Clone / download this repo.
2. Create and activate new virtual environment with the required Python version. 

    Alternatively, you may also install it in your existing environment, but this is not recommended as some of your existing packages might change.

    #### Using conda
    ##### Check current Python version
    In the conda terminal (can be accessed via Jupyter notebook / VS code ipython by using `! <TERMINAL CODE>`), type:
    ```
    python --version
    ```    

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

    ##### Create environment
    ```
    python venv "DIRECTORY_TO_YOUR_ENV/ENV_NAME"
    ```

    ##### Activate environment
    ```
    source "DIRECTORY_TO_YOUR_ENV/ENV_NAME/bin/activate"
    ```

3. Install ml_uncertainty.
    
    Navigate to the ml_uncertainty folder from step 1 and in the terminal, type 
    ```
    pip install ml_uncertainty
    ```
    The package should install automatically.


### Testing
To test the package and ensure that it runs correctly, navigate to the package folder and type:
```
pytest
```

If it passes all tests, it means that the package has been correctly installed. 

### Examples
To run the [examples](examples), some additional plots need to be made which require matplotlib and seaborn packages. These can be installed using:
```
pip install matplotlib seaborn
```

Check out these examples to try out the package. These examples are best run in VS code. 
* [Non-linear regression example with a quadratic model](examples/non_linear_regression_quadratic.py)
* [Non-linear regression example with an Arrhenius model](examples/non_linear_regression_arrhenius.py)
* [Parametric model inference with Arrhenius model](examples/parametric_model.py)
* [Error Propagation with Arrhenius model](examples/error_propagation.py)
* [Model inference for a random forest regressor model](examples/ensemble_model.py)





## Benchmarking
`NonLinearRegression`, `ParametricModelInference`, and `ErrorPropagation` classes have been benchmarked against the Python [statsmodels](https://www.statsmodels.org/stable/index.html) package. The codes for this can be found [here](tests/benchmarking/sm_linear_models.py). To run these benchmarking codes, please install statsmodels using:
```
pip install statsmodels==0.14.0
```

The `EnsembleModelInference` does not have a code to benchmark it against to the best of my knowledge. However, it has followed ideas mentioned in various sources such as [Elements of Statistical Learning, Pg 249](https://hastie.su.domains/ElemStatLearn/), this [Stackoverflow answer](https://stats.stackexchange.com/questions/56895/do-the-predictions-of-a-random-forest-model-have-a-prediction-interval), and a few others.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
