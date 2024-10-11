# Earthquake Hazard Prediction System

This system is designed to predict the earthquake for future forecasts, with the location and the type of earthquake, with details of the magnitude, its frequency, and other relevant parameters. This project demonstrates the application of machine learning techniques to predict earthquakes based on historical seismic data. This repository contains the initial implementation phase, ready for deployment and further enhancements.

## **Problem Statement and Approach**
Anticipating seismic tremors is crucial due to their devastating impacts. The project aims to predict the locations and dates of future earthquakes. The potential applications include improving earthquake hazard assessments, which can save lives and reduce infrastructure costs. Using data from USGS Earthquakes, which provides geological locations, magnitudes, and other factors over the past 30 days, we will forecast earthquake occurrences for the next seven days. 

## Overview
Predicting earthquakes has long been a significant challenge due to the highly uncertain conditions of the Earth's crust. Unlike weather forecasting, which has improved significantly with advanced technologies, earthquake prediction remains elusive. However, artificial intelligence offers a promising avenue. By analyzing massive amounts of seismic data, AI can enhance our understanding of earthquakes, anticipate their behavior, and provide quicker and more accurate early warnings. This improvement can aid hazard assessments for infrastructure planning, potentially saving lives and billions of dollars. This project proposes a solution to predict the likely locations of earthquakes in the next seven days. It includes a web application that uses live data from USGS.gov, updated every minute, to forecast potential earthquake locations worldwide in real time.



## **Web app**
You can access the live predictions through this web app [here - To be Added]()

## **Installation**

* Clone the repository:
`$ git clone https://github.com/Rishit605/earthquake-prediction.git`

* Create an environment: (Suggested)
`$ cd earthquake-prediction`
`$ python3 -m venv <<any environment name>>` (If error occurs, download virtual environment for python)
`$ source <<any environment name>>/bin/activate`
`$ pip install --upgrade pip `

* Then install the dependencies via the requirements.txt
`$ pip install -r requirements.txt` (If error occurs in installation, upgrade pip and debug accordingly)

## Usage
Run the application with `$ python -m src.predictions.inference.py` i.e. in the root directory of the project repo.

### Dataset

The dataset is sourced from the United States Geological Survey (USGS) and includes historical seismic data.  The source data is updated in Real-time every minute. Detailed documentation on the dataset and its parameters is provided in the `docs` folder.


### Improvement and conclusion

Though XGboost model has given Higher `roc_auc` and better `recall`, I believe any work given always has some scope for improvement and in here we could also use `RNN or LSTM` for time series or `rather event series forecasting`. LSTMs have hidden memory cells that help in remembering and handeling time series or event series data well. Moreover for xgboost I have just used hyper parameters from already tuned Adaboost models, but we can also tune xgboost hyper parameter and find best parameters using GridSearchCV or RandomSearch.

