<img src="https://www.bertelsmann.com/media/startseite-buehne/bertelsmann-buehne-3200-800_stage_gt_1200_grid.jpg" width="100%">

# Customer Segmentation Project
There are the data provided by Arvato. As a result of the analyzes to be made using these data, the following are targeted:
* Implementation of unsupervised models for Customer and general population segmentation
* mplementation of supervised models to predict future company campaigns and make them more efficient.

## Contents

1. [Installation](#1)
2. [Project Motivation](#2)
3. [Files](#3)
4. [Support files](#4)
5. [Instructions](#5)
6. [Methodology](#6)
7. [Result](#7)
8. [Post of Medium](#8)
9. [Source, Licensing, Authors, and Acknowledgements](#9)
8. [Conclusion](#10)


<a name="1"></a>
# Installation
To run the Jupyter notebooks and python scripts, you will need a standard installation of Anaconda with Python 3.7.x and additional libraries needed on below:

* matplotlib
* seaborn
* H2o
* sklearn
* xgboost
* imblearn
* lightgbm
* xgboost
* catboost

**H2o.XGBootsClassifier** not supported by **Windows OS**. Because of you **must use different platform**. I used **Google-Colab**

<a name="2"></a>
# Project Motivation
In this project, the purpose was to characterize what types of individuals are more likely to be customers of a mail-order retailer and predict which customers would respond positively to marketing campaigns.


<a name="3"></a>
# Files
The information to be used in this project is provided by Arvato for the project. There are 4 data files and 2 information of attributes files associated with this project:
* **`Udacity_AZDIAS_052018.csv`**: <br>Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
* **`Udacity_CUSTOMERS_052018.csv`**: <br>Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
* **`Udacity_MAILOUT_052018_TRAIN.csv`**: <br>Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
* **`Udacity_MAILOUT_052018_TEST.csv`**:<br> Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
* **`DIAS Attributes - Values.xls`**:<br> Gives the meaning of the column names.
* **`DIAS Information Levels - Attributes.xls`**: <br> Gives what the values in each column mean.

**Note**: The data used for this project not publicly available. It was given for a short time, only to the participants in the competition.

<a name="4"></a>
# Support files
In this project, some of the functions used are developed inside the file **utils.py**.

<a name="5"></a>
# Instructions
To make use of the project, you must access to the repository notebooks and execute the commands presented in it. This project uses 3 jupyter notebooks and a python file, which must be executed in the order indicated:
1. ../000_Preprocessing.ipynb             : Contain data preprocessing and feature engineering.
2. ../001_Unsupervised_Learning.ipynb     : Contain Unsupervised learning techniques.
3. ../002_SupervisedLearning.ipynb        : Contain supervised learning models, metrics evaluation, and prediction for Kaggle submission.
4. ../myutils/utils.py                    : Contain Supported file. For project, some of the functions used are developed inside this file.
5. ../Arvato-Report of CS.pdf             : Contain report of this project
6. ../Last/data/kaggle_submission_file.csv: Kaggle Submission file for compedition of predictions

<a name="6"></a>
# Methodology
Analysis process inside the project consists of 4 main sections.

### Data cleaning and preprocessing
In this first section, an initial display of the relevant data and metrics were carried out, their cleaning as well as feature engineering for further steps.

### Population-Customer Segmentation with Unsupervised Learning
Using the Kmean model, creating high-potential customer classes with Unsupervised Learning method within the general population.

### Mailout campaigns forecasting with Supervised Learning
Implementation of supervised models such as Lightgbm, XGBoost, Catboost, Random Forest, Logistic Regression, and finally **VotingClassifier** for the forecasting of future company campaigns seeking to improve their performance.

### Kaggle Competition
This used the chosen model to make predictions on the campaign data as part of a **Kaggle Competition**.

<a name="7"></a>

# Results
* This process is classifier. Because of this, I choose and tried ***Lightgbm, XGBoost, CatBoost, Random Forest, Logistic Regression classifier models** by the library of Sklearn. My best result for **ROC-AUC Score** from these models is **0.80574 from Lightgbm***.
* After new oversampling to data set improved best *ROC-AUC scored* around **0.815448 from Lightgbm** and other one is **0.871796 from VotingClassifier**. Use predicting of the test label using this **VotingClassifier** model.
* Also Mailout-Test results are  {1: 3658, 0 : 33542} with VotingClassifier model.
* **Kaggle** Submissoin files -> '../Last/data/kaggle_submission_file.csv' [Click it](https://github.com/huseyinelci2000/CustomerSegmentation/blob/master/Last/data/kaggle_submission_file.csv)
* _**The detailed analysis** of the results can be read_ in this **[Medium post](https://medium.com/@huseyinelci2000/finding-new-customers-using-machine-learning-f03857c7f965)** or in **Arvato-Report of Customer Segmentation.pdf**


<a name="8"></a>
# Post to Medium
**[Medium post](https://medium.com/@huseyinelci2000/finding-new-customers-using-machine-learning-f03857c7f965)**

<a name="9"></a>
# Source, Licensing, Authors, and Acknowledgements

#### Source and  Licensing
The **dataset** owner is [Bertelsmann-Arvato](https://www.bertelsmann.com/) The data used for this project not publicly available. It was given for a short time, only to the participants in the competition. You may use only software code pages.

#### Authors
Huseyin ELCI <br>
[Github](https://github.com/huseyinelci)  |  [Kaggle](https://www.kaggle.com/huseyinelci)  |  [Linkedin](https://www.linkedin.com/in/huseyinelci/)
#### Acknowledgements
Thanks to **[Bertelsmann-Arvato](https://www.bertelsmann.com/)** for providing cool data with which we can create a cutting edge project.

---
<a name="10"></a>
## Conclusion
* Trained a K-means model on the general-customers population data sets. Used the model to cluster the customer data for the customer segmentation and then was compared distributions of clusters.
* **Stacking** and **Voting** were useful than a **single model** result.
* It would be nice if we present our findings to the customer and receive feedback.
* Two challenges of this project is **the large data size and the data imbalance.** Cleaning of this big data and applying GridSearcheCV to the models also requires serious time and machine performance. For the solution, it should spend some more time to get to know the columns and seek high performance without disabling the important columns.
Implementing GridSearchCV with 10–128 variations for each model was a mistake for me. Was be wiser to focus only on LGBM and XGBoost models.

**It was instructive, it was worth it.** You may touch the code. Have a enjoy. **:)**
