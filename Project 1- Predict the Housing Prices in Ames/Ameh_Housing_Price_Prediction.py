#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[53]:


# imports
import numpy as np
import pandas as pd
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


# In[54]:


# input data
ames_data = pd.read_csv('Ames_data.csv')

dat_file = 'project1_testIDs.dat'
test_ids = np.loadtxt(dat_file).T


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


def transform_xtest(xtrn, ytrn, xtst, ytst):
    
    # get winsorized limits in Gr_Liv_Area
    winsz_Gr_Liv_Area = winsorize(xtrn.Gr_Liv_Area, limits=[0.05, 0.05])
    a = np.array(xtst['Gr_Liv_Area'].values.tolist())
    xtst['Gr_Liv_Area'] = np.where(a > max(winsz_Gr_Liv_Area), max(winsz_Gr_Liv_Area), a).tolist()
    xtst['Gr_Liv_Area'] = np.where(xtst['Gr_Liv_Area'] < min(winsz_Gr_Liv_Area), min(winsz_Gr_Liv_Area), xtst['Gr_Liv_Area']).tolist()
    
    # get winsorized limits in Sale_Price
    winsz_sale_price = winsorize(ytrn, limits=[0.05, 0.05])
    a = np.array(ytst.values.tolist())
    ytst = np.where(a > max(winsz_sale_price), max(winsz_sale_price), a).tolist()
    ytst = np.where(a < min(winsz_sale_price), min(winsz_sale_price), a).tolist()

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


# In[57]:


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
X = ames_data.loc[:, ~ames_data.columns.str.contains('Sale_Price')]

# label encode features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = lbl.fit_transform(X[col].astype(str))
        
j = 4
X_train = X.iloc[~X.index.isin(test_ids[j]),:]
X_test = X.iloc[X.index.isin(test_ids[j]),:]
y_train = ames_data.iloc[~ames_data.index.isin(test_ids[j]),:]['Sale_Price']
y_test = ames_data.iloc[ames_data.index.isin(test_ids[j]),:]['Sale_Price']


# In[58]:


xtrn_transformed = transform_xtrain(X_train, y_train)
xtst_transformed = transform_xtest(xtrn_transformed, y_train, X_test, y_test)
ytrn_transformed = transform_ytrain(y_train)
ytst_transformed = transform_ytest(y_test, y_train)


# In[59]:


X_train = xtrn_transformed
X_test = xtst_transformed
y_train = ytrn_transformed
y_test = ytst_transformed


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


# In[62]:


# for quick testing
# params = {'max_depth': 500,
#  'eta': 0.1,
#  'min_child_weight': 7,
#  'gamma': 10000,
#  'subsample': 0.8,
#  'colsample_bytree': 0.6,
#  'objective': 'reg:squarederror',
#  'reg_alpha': 100}

# num = 60
# param = params
# param['nthread'] = 4
# param['eval_metric'] = 'rmsle'
# evallist = [(dtest, 'eval'), (dtrain, 'train')]
# num_round = num
# bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
# ypred = bst.predict(dtest)
# rmsle = calc_rmsle(ypred, y_test)
# rmsle


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

result_df = pd.DataFrame(columns=['test_id', 'RMSLE', 'Time for training [Sec]'])

for idx in range(10):
    time_before_hyperparameter_tuning = time.time()
    X = ames_data.loc[:, ~ames_data.columns.str.contains('Sale_Price')]

    # label encode features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = lbl.fit_transform(X[col].astype(str))

    X_train = X.iloc[~X.index.isin(test_ids[idx]),:]
    X_test = X.iloc[X.index.isin(test_ids[idx]),:]
    y_train = ames_data.iloc[~ames_data.index.isin(test_ids[idx]),:]['Sale_Price']
    y_test = ames_data.iloc[ames_data.index.isin(test_ids[idx]),:]['Sale_Price']
    
    xtrn_transformed = transform_xtrain(X_train, y_train)
    xtst_transformed = transform_xtest(xtrn_transformed, y_train, X_test, y_test)
    ytrn_transformed = transform_ytrain(y_train)
    ytst_transformed = transform_ytest(y_test, y_train)
    
    X_train = xtrn_transformed
    X_test = xtst_transformed
    y_train = ytrn_transformed
    y_test = ytst_transformed

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
    dtest = xgb.DMatrix(X_test, label=y_test)
    ypred = bst.predict(dtest)
    rmsle = calc_rmsle(ypred, y_test)
    
    time_after_hyperparameter_turing = time.time()
    time_for_runnig = time_after_hyperparameter_turing - time_before_hyperparameter_tuning
    
    print(f'xgboost RMSLE for testId {idx+1} is: {rmsle}')
    
    result_df = result_df.append({'test_id':idx+1, 'RMSLE':rmsle, 'Time for training [Sec]':
                                 time_for_runnig}, ignore_index=True)
    
    # Save files to csv
    data = {'PID': test_ids[idx], 'True Sale Price': y_test}
    pd.DataFrame.from_dict(data).to_csv(f'./test_y/test_y_for_testID_{idx+1}.csv')
    
    X_test['PID'] = test_ids[idx]
    X_test.to_csv(f'./test/test_for_testID_{idx+1}.csv')
    
    X_train['Sale_Price'] = y_train
    X_train['PID'] = ames_data.iloc[~ames_data.index.isin(test_ids[idx]),:]['PID']
    X_train.to_csv(f'./train/train_for_testID_{idx+1}.csv')
    
    data = {'PID': test_ids[idx], 'Predicted Sale Price': ypred}
    pd.DataFrame.from_dict(data).to_csv(f'./predicted_Sale_Price_for_test/pred_y_for_testID_{idx+1}.csv')


# In[65]:


result_df.to_csv("result.csv")

