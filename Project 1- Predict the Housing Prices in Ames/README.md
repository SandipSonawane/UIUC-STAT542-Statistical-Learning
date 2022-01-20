## 1. Introduction
The goal of this project is to predict the final price of a home (in log scale) using the Ames housing dataset available on Kaggle. This is primarily a regression problem.

In this project preprocessing of input data is done to make sure our model does not deviate too much because of outliers or missing values. Furthermore, the explanatory variables that do not have a correlation with our target variable, will be dropped before they are fed to the model.

Two models will be built:
- using linear regression with Lasso or Ridge or Elasticnet penalty
- using tree-based models such as random forest or boosting trees.
Hyperparameter tuning is performed to ensure our models attain minimum root means squared log error.

The tuned models perform well beating the benchmark scores by more than 10%.

## 2. Data Source
The data is taken from Kaggle. There are 83 columns in the input data file with 2930 row items. There are 10 different test id lists, each containing 879 objects. Out of 83 columns, 37 columns are numeric whereas 46 columns are categorical. The target variable here is Sale_Price. The description of each variable can be found at this link.

## 3. Evaluation metric
To measure the performance of regression models, root means squared log error is used as a performance metric.

## 4. Data Pre-processing
This step ensures that we remove the unwanted data from our training data and allow the models to have data that is clean and has valuable information to generate model parameters. Pre-processing is performed both on categorical variables as well as numerical variables. The Sale Price is winsorized at [0.05, 0.95] to limit the effect of outliers.
### 4.1 Handling Categorical Variables
Below variables are selected that have different mean sale prices for its sub-categories. An example of a selected variable (left side figure) and a rejected variable (right side figure) can be found in the below figure.

> selected_cat_col = ['Sale_Type', 'Fireplace_Qu', 'Kitchen_Qual', 'Central_Air', 'Heating_QC', 'Bsmt_Exposure', 'Bsmt_Qual', 'Exter_Qual', 'Overall_Qual', 'Neighborhood', 'MS_Zoning', 'Garage_Type', 'Sale_Condition',  'Paved_Drive', 'Garage_Finish', 'MS_SubClass', 'Electrical','Foundation', 'condition_1']
### 4.2 Handling Numerical Variables
Most of the variables are not correlated with the target variable. Hence, the variables that have less than 0.3 correlation with the target variable are dropped. Below variables are then selected.

<img src="correlationPlot_final.png" width="50%">

>['Mas_Vnr_Area', 'Total_Bsmt_SF', 'Gr_Liv_Area', 'Full_Bath', 'TotRms_AbvGrd', 'Fireplaces', 'Garage_Cars', 'Wood_Deck_SF',  'Open_Porch_SF', 'Sale_Price', 'Age_of_Property']


Variables PID, Mo_Sold, Year_Sold are dropped as they are not having any relationship with the sale price.

Year_built and year_remod_add are related and they have the same relation with the sale price. Hence, year_remod_add is dropped and year_built is replaced with the age of the property by minimizing the year built with the year 2010.

## 5. Modeling
The dataset is split into training and testing sets The training dataset is further split into estimation and validation sets.
### 5.1 Tree-based model
To build a tree-based model, we used a very famous model which is known as Xgboost. We used the python version. Categorical variables were first encoded in numbers using sklearn. The base model with minimal parameters produced an RMSLE of 0.1779 which was slightly above the benchmark set.

We used the grid search hyperparameter tuning approach. We generated a grid with the below parameters and selected the model parameters that produced the best RMSLE.
```
params = { 'max_depth': [500], # specifies the max depth of a tree
        'eta': [0.1, 0.15, 0.2, 0.25], # Step size shrinkage used in update to prevents overfitting
        'min_child_weight': [5, 7], # Minimum sum of instance weight (hessian) needed in a child.
        'gamma': [200, 2000, 10000], # Min. loss reduction required to make a further partition on a leaf node of the tree.
        'subsample': [0.8], # Setting it to 0.8 means that XGBoost would randomly sample 80% of the training data prior to growing trees. and this will prevent overfitting.
        'colsample_bytree': [0.6, 0.8, 1.0], # parameter for subsampling of columns.
        'objective': ['reg:squarederror'],  # our objective function
        'reg_alpha':[50, 100] # L1 regularization term on weights. }
```

After extensive hyperparameters tuning, the below parameters gave the best RMSLE.

```
{'max_depth': 500, 'eta': 0.1,  'min_child_weight': 7,  'gamma': 10000,  'subsample': 0.8,  'colsample_bytree': 0.8,  'objective': 'reg:squarederror', 'reg_alpha': 100, 'nthread': 4,  'eval_metric': 'rmsle'}  # here nthread=4  means 4 parallel threads were used to run XgBoost.
```

### 5.2 Least-Squares based Linear model
To develop the linear model, we used three modeling paradigms. Lasso, Ridge and Elastic net. We developed a function to trim the number of dependent variables based on collinearity. We created categorical and numerical data pipelines to facilitate the quick transformation of the training and test test. Post which we used 10 fold cross validation to determine the optimal value of lambda (alpha in case of scikit learn). In the case of elastic net there was an additional parameter called l1_ratio. We determined the optimum value of the additional hyperparameter using 10 fold cross validation. The entire training time across all three algorithms have been recorded and saved in csv files (namely lasso_result.csv, ridge_result.csv, elastic_result.csv)

## 6. Running Time
The system used for running xgboost model: Legion 5, Win 11, Intel Core i7-10750H CPU @ 2.60GHz. 16 GB ram. It took 46 seconds to tune the hyperparameters for xgboost. Four computer cores were used for parallel processing.

The system used for running the Linear Models (Ridge, Lasso, and elastic net): Win 10, Intel Core i5 - 10600K CPU @4.10GHz (12CPUs). The training time is summarized in the table above
## 7. Interesting Findings
Through this project, we found below findings:
We can remove most variables and still achieve decent performance.
If we feed all variables to train the xgboost model, the time required for hyperparameter tuning is almost double (86 seconds.)
## 8. Conclusion
The performance of the xgboost model heavily depends on hyperparameter tuning, outliers in the data, especially in the target variable. In this project we used grid search-based hyperparameter tuning for the xgboost model, to reduce tuning time, alternative search algorithms such as random grid-search/ stochastic grid search can be used.
