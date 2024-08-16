<h1 align="center">White Wine Quality Prediction using Python and SciKitLearn</h1>

[Repository Link](https://github.com/MarkStocksUK/White_Wine_Analysis)

## Executive Summary
This project aimed to improve the quality of white wine by developing a predictive model that scores the quality based on physicochemical inputs. Using data from over four thousand Vinho Verde wines, I used Google Colab, Python, SciKit-Learn, pandas, NumPy, seaborn, and matplotlib to clean, analyse and explore the data.

I then compared the performance of four models: Logistical Regression, Decision Tree Classification, Random Forest Classification and Extra Trees Classification. The model can now take the physicochemical composition of the wine and predict whether it is “good”, “average” or “bad” with an accuracy of 98%.

## Data Preprocessing

### Data Source
The dataset has been available for 15 years and contains data on 9999 Portuguese wines. I am focussing on the white wine data, as less work has been done with this data
[link to dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). The data was downloaded as a CSV, imported into Google Colab and then loaded into a Pandas dataframe using the `pd.read.csv` command.

### Initial Exploration
#### Data Information
`wwdata.info()`
The data consists of 4897 rows and 12 variables, 11 input variables and 1 output variable (Quality).  
  
![File information](images/Data_Preprocessing/file_info.jpg)

#### Descriptive Statistics
`wwdata.describe()` Some big variances in the min and max for fields such as *residual sugar*, *free sulfur dioxide* and *total sulfur dioxide* 

![Descriptive Statistics](images/Data_Preprocessing/descriptive_stats.jpg)
 

    ### Missing and null values

    ### Identifying and Handling Outliers

### Investigating the Quality column

  ## Data Analysis
    ### Toolkit
    ### Correlation
    ### Banding column
    ### Handling unbalanced data
    ### Methods

  ## Results
