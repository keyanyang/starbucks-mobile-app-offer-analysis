Starbucks Mobile App Offer Analysis
==============================

Build machine learning models to predict whether or not someone will respond to a Starbucks mobile app offer.

Introduction
------------

Once every few days, Starbucks sends out an offer to users of the mobile app. Not all users receive the same offer and not all users take advantage of the offer they were given. The data set used for this article contains data that mimics customer behavior on the Starbucks rewards mobile app. 

I build models to predict whether a customer will make use of the offer and dig into the data to find insights about Starbucks mobile app offer.

Installation
------------

To run this project, you need to install the required Python packages. You can find them in the requirements file. The easy way to do it is to use the command below.


Commands
------------

Note that the virtual environment has to be active to run the command below.

`make requirements` to install Python dependencieis

`make db` to create an SQL database in data/raw using csv.gz files from data/raw

`make clean` to remove all compiled Python files

`make lint` to check the source code using flake8

Jupyter Notebook
------------

[**➠   Go to the analysis in the Jupyter Notebook**](notebooks/analysis.ipynb)


Post
------------
[Starbucks Mobile App: How to Send Right Offer to Right Person](https://www.keyanyang.com/data/2020/07/09/starbucks-mobile-app-how-to-send-right-offer-to-right-person)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make db`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
