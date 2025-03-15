# Release 0.1.1

0.1.1 (2025-03-07)
------------------
* Cleaned up the repository by removing unnecessary files.
* Added ML uncertainty logo to README.md
* Converted old examples to ipynb format. Deleted corresponding .py files
* Added new examples illustrating ML uncertainty for basis spline functions
* Added a new example to showcase a simple 1D linear regression
* Changed the set_error_dof function in ParametricModelInference to match with the logic of Statsmodels. Ensured correct degrees of freedom for cases with and without intercept in linear regression. Minor changes to documentation and linting for DOF feature
* Removed support for Python 3.12