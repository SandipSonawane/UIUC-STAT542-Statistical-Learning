# imports
import numpy as np
import pandas as pd
import os
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.stats import norm, skew
from scipy.stats import shapiro
import pylab
plt.style.use('ggplot')
from scipy.stats.mstats import winsorize
import time
#from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[54]:


# input data
# ames_data = pd.read_csv('Ames_data.csv')
#
# dat_file = 'project1_testIDs.dat'
# test_ids = np.loadtxt(dat_file).T

from sklearn import preprocessing

trainCSV = pd.read_csv('train.csv')
testCSV = pd.read_csv('test.csv')

# label encode training data
lbl = preprocessing.LabelEncoder()
X = trainCSV.loc[:, ~trainCSV.columns.str.contains('Sale_Price')]
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = lbl.fit_transform(X[col].astype(str))

X_train_input = X

# label encode test data
lbl = preprocessing.LabelEncoder()
X = testCSV
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = lbl.fit_transform(X[col].astype(str))

X_test_for_evaluation = X

y_train_input = trainCSV['Sale_Price']


# ## Preprocess data

# In[55]:


# helper functions
def get_df_info(input_of):
    df_info = pd.DataFrame(columns=['column', 'Null Count', 'Data Type'])
    for col in input_of:
        Null_count  = sum(pd.isnull(input_of[col]))
        dtype = input_of[col].dtype
        df_info = df_info.append({'column': col, 'Null Count': Null_count, 'Data Type': dtype},
                                ignore_index = True)
    
    return df_info

def calc_rmsle(ypred, ytest):
    return np.sqrt(np.mean((np.log(ypred) - np.log(ytest))**2))


# In[56]:


# pre-process train data

def select_column_list(X, target):
    dataframe = X
    dataframe['Sale_Price'] = target
    # Selected Categorical Variables: 
    selected_cat_col = ['Sale_Type', 'Fireplace_Qu', 'Kitchen_Qual', 'Central_Air', 'Heating_QC', 'Bsmt_Exposure', 
                         'Bsmt_Qual', 'Exter_Qual', 'Overall_Qual', 'Neighborhood', 'MS_Zoning', 'Garage_Type', 'Sale_Condition', 
                         'Paved_Drive', 'Garage_Finish', 'MS_SubClass', 'Electrical','Foundation', 'condition_1']
    
    # select numerical variables based on correlation
    df_info = get_df_info(dataframe)
    num_vars = df_info[df_info['Data Type'] != 'object']
    ames_num_cols_df = dataframe.loc[:, dataframe.columns.isin(num_vars.column.tolist())]
    corr_df = ames_num_cols_df.corr()
    corr_df_shortlisted = corr_df.loc[(corr_df['Sale_Price'] > 0.3) | (corr_df['Sale_Price'] < -0.3),:]
    corr_df_shortlisted = corr_df_shortlisted.loc[:, corr_df_shortlisted.columns.isin(corr_df_shortlisted.index)]
    corr_df_shortlisted = corr_df_shortlisted.loc[~corr_df_shortlisted.index.isin(['Garage_Yr_Blt']), ~corr_df_shortlisted.columns.isin(['Garage_Yr_Blt'])]
    corr_df_shortlisted = corr_df_shortlisted.loc[~corr_df_shortlisted.index.isin(['Garage_Area', 'First_Flr_SF']), ~corr_df_shortlisted.columns.isin(['Garage_Area', 'First_Flr_SF'])]
    
    # finalize columns selected
    selected_cat_col.extend(corr_df_shortlisted.columns) 
    
    return selected_cat_col


def transform_xtrain(xtrn, ytrn):
    
    # winsorize outliers in Gr_Liv_Area
    winsz_Gr_Liv_Area = winsorize(xtrn.Gr_Liv_Area, limits=[0.05, 0.05])
    xtrn['Gr_Liv_Area'] = winsz_Gr_Liv_Area
    
    winsz_sale_price = winsorize(ytrn, limits=[0.05, 0.05])
    ytrn = winsz_sale_price

    # drop PID since this won't affect Sale_Price
    xtrn.drop(['PID'], axis=1, inplace=True)
    
    # Convert Year values to age and drop unnecessary columns
    xtrn.drop(['Year_Remod_Add', 'Mo_Sold', 'Year_Sold'], axis=1, inplace=True)
    xtrn['Age_of_Property'] = 2010 - xtrn['Year_Built']
    xtrn.drop(['Year_Built'], axis=1, inplace=True)                                            
    
    selected_cat_col = select_column_list(xtrn, ytrn)
    df_final = xtrn.loc[:, xtrn.columns.isin(selected_cat_col)]
    df_final.drop(['Sale_Price'], axis = 1, inplace = True)
    
    return df_final


def transform_ytrain(ytrn):
    winsz_sale_price = winsorize(ytrn, limits=[0.05, 0.05])
    ytrn = winsz_sale_price
    
    return ytrn


def transform_xtest(xtrn, ytrn, xtst):
    
    # get winsorized limits in Gr_Liv_Area
    winsz_Gr_Liv_Area = winsorize(xtrn.Gr_Liv_Area, limits=[0.05, 0.05])
    a = np.array(xtst['Gr_Liv_Area'].values.tolist())
    xtst['Gr_Liv_Area'] = np.where(a > max(winsz_Gr_Liv_Area), max(winsz_Gr_Liv_Area), a).tolist()
    xtst['Gr_Liv_Area'] = np.where(xtst['Gr_Liv_Area'] < min(winsz_Gr_Liv_Area), min(winsz_Gr_Liv_Area), xtst['Gr_Liv_Area']).tolist()
    
    # get winsorized limits in Sale_Price
    winsz_sale_price = winsorize(ytrn, limits=[0.05, 0.05])
    # a = np.array(ytst.values.tolist())
    # ytst = np.where(a > max(winsz_sale_price), max(winsz_sale_price), a).tolist()
    # ytst = np.where(a < min(winsz_sale_price), min(winsz_sale_price), a).tolist()

#     # drop PID since this won't affect Sale_Price
#     xtst.drop(['PID'], axis=1, inplace=True)
    
    # Convert Year values to age and drop unnecessary columns
    xtst.drop(['Year_Remod_Add', 'Mo_Sold', 'Year_Sold'], axis=1, inplace=True)
    xtst['Age_of_Property'] = 2010 - xtst['Year_Built']
    xtst.drop(['Year_Built'], axis=1, inplace=True)
    
#     selected_cat_col = select_column_list(xtrn, ytrn)
    xtst_final = xtst.loc[:, xtst.columns.isin(xtrn.columns.tolist())]
    
    return xtst_final


def transform_ytest(ytst, ytrn):
    # get winsorized limits in Sale_Price
    winsz_sale_price = winsorize(ytrn, limits=[0.05, 0.05])
    a = np.array(ytst.values.tolist())
    ytst = np.where(a > max(winsz_sale_price), max(winsz_sale_price), a).tolist()
    ytst = np.where(ytst < min(winsz_sale_price), min(winsz_sale_price), ytst).tolist()
    
    return ytst


# Transform data

# In[58]:
from sklearn.model_selection import train_test_split
# X_train_input, y_train_input = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(
    X_train_input, y_train_input, test_size=0.20, random_state=42)

xtrn_transformed = transform_xtrain(X_train, y_train)
xtst_transformed = transform_xtest(xtrn_transformed, y_train, X_test)
X_test_for_evaluation = transform_xtest(xtrn_transformed, y_train, X_test_for_evaluation)
ytrn_transformed = transform_ytrain(y_train)
ytst_transformed = transform_ytest(y_test, y_train)

# In[59]:

X_train = xtrn_transformed
X_test = xtst_transformed
y_train = ytrn_transformed
y_test = ytst_transformed

# In[57]:


# ## XgBoost Model

# ### Hyperparameters tuning

# In[60]:


# Create a list of hyperparameters
# ref: https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python
from itertools import product

def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

params = {
        'max_depth': [500],
        'eta': [0.1, 0.15, 0.2, 0.25],
        'min_child_weight': [5, 7],
        'gamma': [200, 2000, 10000],
        'subsample': [0.8],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'objective': ['reg:squarederror'],
        'reg_alpha':[50, 100]
        }
params_df = expand_grid(params)


# In[61]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)




# In[63]:


time_before_hyperparameter_tuning = time.time()
rmsles = {}

for num in [60]:
    for idx in params_df.index:
        param = params_df.iloc[idx,:].to_dict()
        param['nthread'] = 4
        param['eval_metric'] = 'rmsle'
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = num
        bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
        ypred = bst.predict(dtest)
        rmsle = calc_rmsle(ypred, y_test)
        rmsles.update({rmsle:param})
        
time_after_hyperparameter_turing = time.time()
print(f'time for hyperparameter tuning: {time_after_hyperparameter_turing - time_before_hyperparameter_tuning}')

print('min rmsle achieved: ', min(rmsles.keys()))
print('tuned parameters: \n', rmsles[min(rmsles.keys())])


# In[64]:


param = rmsles[min(rmsles.keys())]
param['nthread'] = 4
param['eval_metric'] = 'rmsle'

num_round = num


# Predict sale price on test data and create submission files.
result_df = pd.DataFrame(columns=['test_id', 'RMSLE', 'Time for training [Sec]'])

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

evallist = [(dtest, 'eval'), (dtrain, 'train')]
# final model with best parameters
bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)






xtst_transformed = transform_xtest(xtrn_transformed, y_train, X_test)

dtest = xgb.DMatrix(X_test_for_evaluation)
ypred = bst.predict(dtest)
ypred = transform_ytest(ypred, y_train)

ypred


# for idx in range(10):
#     time_before_hyperparameter_tuning = time.time()
#     X = ames_data.loc[:, ~ames_data.columns.str.contains('Sale_Price')]
#
#     # label encode features
#     for col in X.columns:
#         if X[col].dtype == 'object':
#             X[col] = lbl.fit_transform(X[col].astype(str))
#
#     X_train = X.iloc[~X.index.isin(test_ids[idx]),:]
#     X_test = X.iloc[X.index.isin(test_ids[idx]),:]
#     y_train = ames_data.iloc[~ames_data.index.isin(test_ids[idx]),:]['Sale_Price']
#     y_test = ames_data.iloc[ames_data.index.isin(test_ids[idx]),:]['Sale_Price']
#
#     xtrn_transformed = transform_xtrain(X_train, y_train)
#     xtst_transformed = transform_xtest(xtrn_transformed, y_train, X_test, y_test)
#     ytrn_transformed = transform_ytrain(y_train)
#     ytst_transformed = transform_ytest(y_test, y_train)
#
#     X_train = xtrn_transformed
#     X_test = xtst_transformed
#     y_train = ytrn_transformed
#     y_test = ytst_transformed
#
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtest = xgb.DMatrix(X_test, label=y_test)
#
#     evallist = [(dtest, 'eval'), (dtrain, 'train')]
#     bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
#     dtest = xgb.DMatrix(X_test, label=y_test)
#     ypred = bst.predict(dtest)
#     rmsle = calc_rmsle(ypred, y_test)
#
#     time_after_hyperparameter_turing = time.time()
#     time_for_runnig = time_after_hyperparameter_turing - time_before_hyperparameter_tuning
#
#     print(f'xgboost RMSLE for testId {idx+1} is: {rmsle}')
#
#     result_df = result_df.append({'test_id':idx+1, 'RMSLE':rmsle, 'Time for training [Sec]':
#                                  time_for_runnig}, ignore_index=True)
#
#     # Save files to csv
#     data = {'PID': test_ids[idx], 'True Sale Price': y_test}
#     pd.DataFrame.from_dict(data).to_csv(f'./test_y/test_y_for_testID_{idx+1}.csv')
#
#     X_test['PID'] = test_ids[idx]
#     X_test.to_csv(f'./test/test_for_testID_{idx+1}.csv')
#
#     X_train['Sale_Price'] = y_train
#     X_train['PID'] = ames_data.iloc[~ames_data.index.isin(test_ids[idx]),:]['PID']
#     X_train.to_csv(f'./train/train_for_testID_{idx+1}.csv')
#
#     data = {'PID': test_ids[idx], 'Predicted Sale Price': ypred}
#     pd.DataFrame.from_dict(data).to_csv(f'./predicted_Sale_Price_for_test/pred_y_for_testID_{idx+1}.csv')


# In[65]:

#
# # result_df.to_csv("result.csv")
#
# ## Linear Models Section
#
# # Importing all the relevant mdoules
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RepeatedKFold
# import time
# import pandas as pd
# import numpy as np
#
#
# # Helper function below to select columns
# def select_column_list(X, target):
#     dataframe = X
#     dataframe['Sale_Price'] = target
#     # Selected Categorical Variables:
#     selected_cat_col = ['Sale_Type', 'Fireplace_Qu', 'Kitchen_Qual', 'Central_Air', 'Heating_QC', 'Bsmt_Exposure',
#                          'Bsmt_Qual', 'Exter_Qual', 'Overall_Qual', 'Neighborhood', 'MS_Zoning', 'Garage_Type', 'Sale_Condition',
#                          'Paved_Drive', 'Garage_Finish', 'MS_SubClass', 'Electrical','Foundation', 'condition_1']
#
#     # select numerical variables based on correlation
#     df_info = get_df_info(dataframe)
#     num_vars = df_info[df_info['Data Type'] != 'object']
#     ames_num_cols_df = dataframe.loc[:, dataframe.columns.isin(num_vars.column.tolist())]
#     corr_df = ames_num_cols_df.corr()
#     corr_df_shortlisted = corr_df.loc[(corr_df['Sale_Price'] > 0.3) | (corr_df['Sale_Price'] < -0.3),:]
#     corr_df_shortlisted = corr_df_shortlisted.loc[:, corr_df_shortlisted.columns.isin(corr_df_shortlisted.index)]
#     corr_df_shortlisted = corr_df_shortlisted.loc[~corr_df_shortlisted.index.isin(['Garage_Yr_Blt']), ~corr_df_shortlisted.columns.isin(['Garage_Yr_Blt'])]
#     corr_df_shortlisted = corr_df_shortlisted.loc[~corr_df_shortlisted.index.isin(['Garage_Area', 'First_Flr_SF']), ~corr_df_shortlisted.columns.isin(['Garage_Area', 'First_Flr_SF'])]
#
#     # finalize columns selected
#     selected_cat_col.extend(corr_df_shortlisted.columns)
#
#     return selected_cat_col
#
# # Reading raw data and only keeping the columns that are required
# ames_data = pd.read_csv('Ames_data.csv')
# ind = [col for col in ames_data.columns if col != 'Sale_Price']
# X = ames_data.loc[:,ind]
# y = ames_data.loc[:,'Sale_Price']
# selected_cat_col = select_column_list(X, y)
# ames_final = ames_data.loc[:, ames_data.columns.isin(selected_cat_col)]
#
# # Separating out all the categorical and numerical attributes
# num_attributes = []
# cat_attributes = []
#
# for col in ames_final.columns:
#     if ames_final[col].dtype == 'object':
#         cat_attributes.append(col)
#     elif ames_final[col].dtype != 'object' and col != 'Sale_Price':
#         num_attributes.append(col)
#
# # Creating a helper class to select columns from the data frame
# class DataFrameSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, attribute_names):
#         self.attribute_names = attribute_names
#
#     def fit(self, X, y = None):
#         return self
#
#     def transform(self,X):
#         return X[self.attribute_names].values
#
#  # Creating categorical pipelines
# cat_pipeline = Pipeline([
#     ('selector', DataFrameSelector(cat_attributes)),
#     ('ordinal_encoder', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1))
# ])
#
#
# # creating numerical pipelines
# num_pipeline = Pipeline([
#     ('selector', DataFrameSelector(num_attributes)),
#     ('imputer', SimpleImputer(strategy = 'median')),
#     ('std_scaler', StandardScaler())
# ])
#
# # Combining the categorical and numerical pipeline in one
#
# full_pipeline = ColumnTransformer([
#     ("num_pipeline", num_pipeline, num_attributes),
#     ("cat_pipeline", cat_pipeline, cat_attributes),
# ])
#
#
# # Helper function to calculate rmsle
# def calc_rmsle(ypred, ytest):
#     return np.sqrt(np.mean((np.log(ypred) - np.log(ytest))**2))
#
#
#  # Running the code for each test id
# ridge_dict = {}
# lasso_dict = {}
# elastic_net_dict = {}
#
# for set_id in range(0,10):
#
#     print(set_id)
#
#     ## Sampling out a particular data set for testing purpose
#     X_train = X.iloc[~X.index.isin(test_ids[set_id]),:]
#     X_test = X.iloc[X.index.isin(test_ids[set_id]),:]
#     y_train = ames_final.iloc[~ames_final.index.isin(test_ids[set_id]),:]['Sale_Price']
#     y_test = ames_final.iloc[ames_final.index.isin(test_ids[set_id]),:]['Sale_Price']
#
#     # transforming X and winsorizing Y
#     train_prepared = full_pipeline.fit_transform(X_train)
#     xtest_prepared = full_pipeline.transform(X_test)
#     winsz_sale_price = winsorize(y_train, limits=[0.05, 0.05])
#     y_train = winsz_sale_price
#
#     ############################################ Ridge Regression (Cross Validation and training) ############################################
#     ridge_time_start = time.time()
#
#     # Cross Validation of Ridge Regression
#     ridge_reg = Ridge(solver = "cholesky")
#     cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
#     grid = dict()
#     grid['alpha'] = np.logspace(-1.5, 1.5,20)
#     search = GridSearchCV(ridge_reg, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#     results = search.fit(train_prepared, y_train)
#
#     ## Training the model with the lowest cross validation error
#
#     ridge_mod_fin = Ridge(alpha = results.best_params_['alpha'], solver = "cholesky")
#
#     ridge_mod_fin.fit(train_prepared, y_train)
#     ypred = ridge_mod_fin.predict(xtest_prepared)
#     ridge_rmsle = calc_rmsle(ypred, y_test)
#
#     ridge_time_end = time.time()
#     ridge_dict[set_id] = ((ridge_time_end - ridge_time_start), ridge_rmsle)
#
#     ############################################ Lasso Regression (Cross Validation and training) ############################################
#     lasso_time_start = time.time()
#
#     lasso_reg = Lasso()
#     lasso_cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
#     grid_lasso = dict()
#     grid_lasso['alpha'] = np.logspace(-1.5, 1.5,20)
#     lasso_search = GridSearchCV(lasso_reg, grid_lasso, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#     results_lasso = search.fit(train_prepared, y_train)
#
#     lasso_mod_fin = Lasso(alpha = results_lasso.best_params_['alpha'])
#     #xtest_prepared = full_pipeline.transform(X_test)
#     lasso_mod_fin.fit(train_prepared, y_train)
#     ypred = lasso_mod_fin.predict(xtest_prepared)
#     lasso_rmsle = calc_rmsle(ypred, y_test)
#
#     lasso_time_end = time.time()
#     lasso_dict[set_id] = ((lasso_time_end - lasso_time_start), lasso_rmsle)
#
#     ############################################ Elastic Net (Cross Validation and training) ############################################
#
#     elastic_time_start = time.time()
#
#     elastic_net = ElasticNet()
#     elastic_net_cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
#     elastic_grid = dict()
#     elastic_grid['alpha'] = np.logspace(-1.5, 1.5,20)
#     elastic_grid['l1_ratio'] = np.arange(0, 1, 0.05)
#     elastic_search = GridSearchCV(elastic_net, elastic_grid,scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#     results_elastic_search = elastic_search.fit(train_prepared, y_train)
#
#     elastic_net_mod_fin = ElasticNet(alpha = results_elastic_search.best_params_['alpha'], l1_ratio = results_elastic_search.best_params_['l1_ratio'])
#     elastic_net_mod_fin.fit(train_prepared, y_train)
#     ypred = elastic_net_mod_fin.predict(xtest_prepared)
#     elastic_net_rmsle = calc_rmsle(ypred, y_test)
#
#     elastic_time_end = time.time()
#     elastic_net_dict[set_id] = ((elastic_time_end - elastic_time_start), elastic_net_rmsle)
#
#
#  ## Writing the results in a csv file
# lasso_res =  pd.DataFrame.from_dict(lasso_dict, orient = 'index').reset_index().rename(columns = {'index':'test_id', 0:'time', 1: 'RMSLE'})
# ridge_res =  pd.DataFrame.from_dict(ridge_dict, orient = 'index').reset_index().rename(columns = {'index':'test_id', 0:'time', 1: 'RMSLE'})
# elastic_net_res =  pd.DataFrame.from_dict(elastic_net_dict, orient = 'index').reset_index().rename(columns = {'index':'test_id', 0:'time', 1: 'RMSLE'})
#
# lasso_res.to_csv("lasso_result.csv")
# ridge_res.to_csv("ridge_result.csv")
# elastic_net_res.to_csv("elastic_result.csv")