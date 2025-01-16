Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

### Report Bugs


Report bugs at https://github.com/architdatar/ml_uncertainty/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the Boards for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Some new feature ideas can also be found [here](/docs/feature_ideas.md).

### Write Documentation

ML Uncertainty Quantification could always use more documentation, whether as part of the
official ML Uncertainty Quantification docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Enhance the theory

If there are any enhancements / corrections to be made in the theory used, which you can read [here](/docs/theory/), please [report them as a bug](#report-bugs).

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/architdatar/ml_uncertainty/issues.

The best way to send feedback is to file an issue as shown for [bugs](#report-bugs).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `ml_uncertainty` for local development.

1. Fork the `ml_uncertainty` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/ml_uncertainty.git
3. Create a virtual environment to develop this project. Using one of the supported Python versions (see above), this is how you set up your fork for local development:

    ```
    pip install virtualenv==20.25.1
    virtualenv VIRTUAL_ENV_PATH # Including name
    source VIRTUAL_ENV_PATH/bin/activate
    cd ml_uncertainty/
    pip install -r requirements_dev.txt
    ```

For development in Windows, the process is the same as above, except, to activate the
virtual environment, you must use the following:
```
source VIRTUAL_ENV_PATH/Scripts/activate
```

**NOTE for Windows developers:** 

Windows uses CRLF line endings while Unix uses LF by default.
Git tries to match line endings to the OS you are developing on. This 
means that for those developing on Windows, Git might change line endings to CRLF.
This may cause problems when trying to merge with the original code. So, please make
sure that the auto conversion to CRLF is turned off and we force the line endings to
be LF only. 

To ensure this, in your terminal, navigate to the `ml_uncertainty` directory and type this:

```
git config --global core.autocrlf false # To ensure that the existing line endings aren't changed.
git config core.eol lf # To ensure that the line endings in this repo are set to lf.
```
For further reference and details, please refer this [Stackoverflow link](https://stackoverflow.com/questions/2517190/how-do-i-force-git-to-use-lf-instead-of-crlf-under-windows).

4. Create a branch for local development::
    ```
    git checkout -b name-of-your-bugfix-or-feature
    ```
   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the black, flake8, and pytests implemented in tox. This also includes testing other Python versions. 
    ```
    tox
    ```

    TIP: Keep checking your code intermittently with pytest, black, and flake8 to make sure that it is correct. 
    ```
    pytest
    black --check ml_uncertainty tests examples
    flake8 ml_uncertainty tests examples
    ```

6. Commit your changes and push your branch:
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    ```
    Ensure that your branch is in sync with the latest version of the main branch.
    <!-- TODO: Demand that contributors rebase their features from the latest on the main branch -->
    ```    
    git pull origin main 
    ```
    Push to your branch
    ```
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
<!-- 3. The pull request should work for Python 3.5, 3.6, 3.7 and 3.8, and for PyPy. Check
   https://travis-ci.com/architdatar/ml_uncertainty/pull_requests
   and make sure that the tests pass for all supported Python versions. -->
3. The pull request should work with all the Python versions listed above. This should be ensured by testing with tox.

<!-- Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.md).
Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass. -->
