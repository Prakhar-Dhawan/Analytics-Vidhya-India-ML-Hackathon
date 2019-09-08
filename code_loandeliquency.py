# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:04:26 2019

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 01:22:36 2019

@author: DELL
"""


from statsmodels.tsa.stattools import acf, pacf
import pandas as pd# -*- coding: utf-8 -*-
from datetime import timedelta 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
from sklearn.metrics import roc_auc_score,confusion_matrix, classification_report 

#from pandas.tools.plotting import autocorrelation_plot

train = pd.read_csv('train.csv')
train.shape
train.info()
train.columns
#train.is_promoted.value_counts()
#train['previous_year_rating'].isna().sum()
#train['education'].isna().sum()


test = pd.read_csv('test.csv')
test.shape
test.info()
test.columns
#test.avg_training_score.value_counts()
#test['previous_year_rating'].isna().sum()
#test['education'].isna().sum()
'''
#EDA
train1 = train.drop(['m13'],axis=1)
train1.shape
train1.info()
train1.columns
train.m13.value_counts()

df = pd.concat([train1,test])
df.info()
df.shape
df.columns

#UNIVARIATE ANALYSIS
df.source.value_counts()
df.financial_institution.value_counts()
#df[col] = pd.to_datetime(df[col])
df['origination_date'] = pd.to_datetime(df['origination_date'])
df.origination_date.value_counts()
#df.source.value_counts()
df.first_payment_date.value_counts()
df.loan_purpose.value_counts()
df.insurance_type.value_counts()




#finding correlation between different columns
wcol = ['m1','m2']

corr = df[wcol].corr()
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.savefig('corr1.jpeg')
corr.style.background_gradient()
plt.show()

df[['m2','m5']].corr()

#VIF CALCULATION
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Break into left and right hand side; y and X
y, X = dmatrices("m13 ~ m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9 + m10 + m11 + m12 ", data=train, return_type="dataframe")

# For each Xi, calculate VIF
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Fit X to y
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)


#histogram for num varibles
print(df.loan_to_value.describe())
#optional for hist
plt.hist(df['loan_term'],bins = np.arange(df['loan_term'].min(),df['loan_term'].max()))
plt.xticks(np.arange(0.5, df['loan_term'].max()+0.5, 1.0))
plt.xlabel('loan_term')


(df['co_borrower_credit_score']).describe()
sns.distplot(np.log(df['unpaid_principal_bal']))
df['insurance_percent'].plot.box(figsize=(16,5))


#log - if outlers
np.sqrt(df[df['insurance_percent']!=0].insurance_percent).plot.box(figsize=(16,5))
sns.distplot((df[df['insurance_percent']!=0].insurance_percent))
#standardized_X = preprocessing.scale(X)

#standard scaler normalization - for normal dist
from sklearn import preprocessing
standardized_X = preprocessing.scale(df['borrower_credit_score'])
sns.distplot(standardized_X)


#min max normalization - black box
x = df[['debt_to_income_ratio']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
sns.distplot(df_normalized)
(df_normalized).plot.box(figsize=(16,5))
df_normalized.describe()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(df['interest_rate'])
sns.distplot(X['interest_rate'])

test_1 = sc.fit_transform(test_leaderboard)
test_1.shape

#df['interest_rate'].plot.box(figsize=(16,5))
df[(np.sqrt(df['borrower_credit_score'])< 24.9)]
(standardized_X < -3).sum()
(df_normalized < 0.129).sum()

'''

#DATA PREPARATION
test['m13'] = np.nan 
#df["D"] = np.nan
df1 = pd.concat([train,test],sort=False)
df1.info()
df1.shape
df1.columns


df1['financial_institution'] = df1['financial_institution'].astype(object)
df1['financial_institution'].value_counts()
df1['financial_institution'][df1['financial_institution']=='OTHER'] = 0 
df1['financial_institution'][df1['financial_institution']=='Browning-Hart'] = 1 
df1['financial_institution'][df1['financial_institution']=='Swanson, Newton and Miller'] = 2
df1['financial_institution'][df1['financial_institution']=='Edwards-Hoffman'] = 3
df1['financial_institution'][df1['financial_institution']=='Martinez, Duffy and Bird'] = 4
df1['financial_institution'][df1['financial_institution']=='Miller, Mcclure and Allen'] = 5
df1['financial_institution'][df1['financial_institution']=='Nicholson Group'] = 6
df1['financial_institution'][df1['financial_institution']=='Turner, Baldwin and Rhodes'] = 7
df1['financial_institution'][df1['financial_institution']=='Suarez Inc'] = 8
df1['financial_institution'][df1['financial_institution']=='Cole, Brooks and Vincent'] = 9
df1['financial_institution'][df1['financial_institution']=='Richards-Walters'] = 10
df1['financial_institution'][df1['financial_institution']=='Taylor, Hunt and Rodriguez'] = 11
df1['financial_institution'][df1['financial_institution']=='Sanchez-Robinson'] = 12
df1['financial_institution'][df1['financial_institution']=='Sanchez, Hays and Wilkerson'] = 13
df1['financial_institution'][df1['financial_institution']=='Romero, Woods and Johnson'] = 14
df1['financial_institution'][df1['financial_institution']=='Thornton-Davis'] = 15
df1['financial_institution'][df1['financial_institution']=='Richardson Ltd'] = 16
df1['financial_institution'][df1['financial_institution']=='Anderson-Taylor'] = 17
df1['financial_institution'][df1['financial_institution']=='Chapman-Mcmahon'] = 18
df1['financial_institution'] = pd.Categorical(df1.financial_institution)

df1.financial_institution.value_counts()

'''
import seaborn as sns
import matplotlib.pyplot as plt
financial_institution_count = df1['financial_institution'].value_counts()
sns.set(style="darkgrid")
sns.barplot(financial_institution_count.index, financial_institution_count.values, alpha=0.9)
plt.title('Frequency Distribution of financial institution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('financial institution', fontsize=12)
plt.show()

labels = df1['financial_institution'].astype('category').cat.categories.tolist()
counts = df1['financial_institution'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()
'''
df1['source'] = df1['source'].astype(object)
df1['source'].value_counts()
df1['source'][df1['source']=='X'] = 0 
df1['source'][df1['source']=='Y'] = 1 
df1['source'][df1['source']=='Z'] = 2
df1['source'] = pd.Categorical(df1.source)
df1['source'].value_counts()

df1['unpaid_principal_bal'] = np.log(df1['unpaid_principal_bal'])

#df[col] = pd.to_datetime(df[col])
#df1['origination_date'] = pd.to_datetime(df1['origination_date'])
#df1['first_payment_date'] = pd.to_datetime(df1['first_payment_date'])


'''df1['first_payment_date'] = pd.DatetimeIndex(df1['first_payment_date']).year
df1['first_payment_date'].value_counts()
df1['origination_date'] = pd.DatetimeIndex(df1['origination_date']).month


df1['payment_origination_date'] = df1['first_payment_date'] - df1['origination_date']
df1['payment_origination_date'].value_counts()


df1.to_csv('dummydf.csv')

df1_onehot = df1.copy()
df1_onehot = pd.get_dummies(df1_onehot, columns=['financial_institution'], prefix = ['financial_institution'])

print(df1_onehot.head())

df1_onehot.info()
df1_onehot.shape
df1_onehot.columns
'''

df1['first_payment_date'] = df1['first_payment_date'].str[3:5]
df1['origination_date'] = df1['origination_date'].str[3:5]
df1['first_payment_date'].value_counts()
df1['origination_date'].value_counts()
df1['first_payment_date'] = df1['first_payment_date'].astype(int)
df1['first_payment_date'].value_counts()
df1['origination_date'] = df1['origination_date'].astype(int)
df1['origination_date'].value_counts()
df1['payment_origination_date'] = df1['first_payment_date'] - df1['origination_date'] 
df1['payment_origination_date'].value_counts()
df1['payment_origination_date'] =  df1['payment_origination_date'].astype(int)

df1['borrower_credit_score'].describe()
df1['borrower_credit_score'][df1['borrower_credit_score']==0] = 0
df1['borrower_credit_score'][(df1['borrower_credit_score']>=300) & (df1['borrower_credit_score']<=669)] =1
df1['borrower_credit_score'][(df1['borrower_credit_score']>=670) & (df1['borrower_credit_score']<=704)] =2
df1['borrower_credit_score'][(df1['borrower_credit_score']>=705) & (df1['borrower_credit_score']<=739)] =3
df1['borrower_credit_score'][(df1['borrower_credit_score']>=740) & (df1['borrower_credit_score']<=759)] =4
df1['borrower_credit_score'][(df1['borrower_credit_score']>=760) & (df1['borrower_credit_score']<=779)] =5
df1['borrower_credit_score'][(df1['borrower_credit_score']>=780) & (df1['borrower_credit_score']<=799)] =6
df1['borrower_credit_score'][(df1['borrower_credit_score']>=800) & (df1['borrower_credit_score']<=824)] =7
df1['borrower_credit_score'][(df1['borrower_credit_score']>=825) & (df1['borrower_credit_score']<=850)] =8
df1['borrower_credit_score'] =  df1['borrower_credit_score'].astype(int)
df1['borrower_credit_score'].value_counts()

df1['co_borrower_credit_score'].describe()
df1['co_borrower_credit_score'][df1['co_borrower_credit_score']==0] = 0
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=300) & (df1['co_borrower_credit_score']<=669)] =1
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=670) & (df1['co_borrower_credit_score']<=739)] =2
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=740) & (df1['co_borrower_credit_score']<=759)] =3
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=760) & (df1['co_borrower_credit_score']<=779)] =4
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=780) & (df1['co_borrower_credit_score']<=799)] =5
df1['co_borrower_credit_score'][(df1['co_borrower_credit_score']>=800) & (df1['co_borrower_credit_score']<=850)] =6
df1['co_borrower_credit_score'] =  df1['co_borrower_credit_score'].astype(int)
df1['co_borrower_credit_score'].value_counts()

df1['loan_purpose'] = df1['loan_purpose'].astype(object)
df1['loan_purpose'].value_counts()
df1['loan_purpose'][df1['loan_purpose']=='A23'] = 0 
df1['loan_purpose'][df1['loan_purpose']=='B12'] = 1 
df1['loan_purpose'][df1['loan_purpose']=='C86'] = 2
df1['loan_purpose'] = pd.Categorical(df1.loan_purpose)

df1['insurance_type'] = pd.Categorical(df1.insurance_type)

df1['minitial'] = df1['m3'] + df1['m4'] + df1['m5']
df1['mintermediate'] = df1['m6'] + df1['m7'] + df1['m8']
df1['mfinal'] = df1['m9'] + df1['m10'] +  df1['m11'] +  df1['m12']


'''
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')
'''

df1.info()
df1.shape
df1.columns

df1 = df1.drop(['origination_date','first_payment_date','number_of_borrowers','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'],axis=1)
#df1.to_csv('test+train_final.csv')


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1[['interest_rate','debt_to_income_ratio']] = sc.fit_transform(df1[['interest_rate','debt_to_income_ratio']])
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
df1[['unpaid_principal_bal','loan_term','loan_to_value','insurance_percent']] = min_max.fit_transform(df1[['unpaid_principal_bal','loan_term','loan_to_value','insurance_percent']])

#df1.to_csv('test+train_final_model.csv')


df_train = df1[df1['m13'].notnull()]
df_test1 = df1[df1['m13'].isna()]

df_train.info()
df_train.shape
df_train.columns


df_test1.info()
df_test1.shape
df_test1.columns

X = df_train.drop(['m13','loan_id'],axis=1)
y = df_train['m13']
df_test = df_test1.drop(['m13','loan_id'],axis=1)

X.info()
X.shape
X.columns

#y = pd.Categorical(y)
y.shape
y.value_counts()


##MODELLING

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# y = dataset.ToPredictField

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=12)
#y = pd.Categorical(y)
#y_train = pd.Categorical(y_train)
#y_test = pd.Categorical(y_test)

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


sm = SMOTE(random_state=12, ratio=0.6)
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
#compare_plot(x_train_res, y_train_res , X_train, y_train, method='SMOTE')


#rus = RandomUnderSampler(random_state=12,ratio=0.009)
#x_train_res1, y_train_res1 = rus.fit_sample(X_train, y_train)

#ros = RandomOverSampler(random_state=12,ratio=0.57)
#x_train_res, y_train_res = ros.fit_sample(x_train_res1, y_train_res1)
y_train.value_counts()
y.value_counts()

x_train_res = pd.DataFrame(x_train_res)
x_train_res.columns = X_test.columns
#X_train.to_csv('X_train.csv')
#x_train_res.to_csv('x_train_res.csv')

#y_train_res = pd.Categorical(y_train_res)
#y.value_counts()
y_train.value_counts()
#y_train_res1.sum()
y_train_res.sum()

#df = pd.concat([train1,test]
X_model = pd.concat([x_train_res,X_test])
#y_model = pd.concat([pd.DataFrame(y_train_res),y_test])
y_model = np.concatenate((y_train_res, y_test))
#y_model = pd.Categorical(y_model)
#y_model.value_counts()
X_model.info()
X_model['source'] = pd.Categorical(X_model['source'])
X_model['financial_institution'] = pd.Categorical(X_model['financial_institution'])
X_model['loan_purpose'] = pd.Categorical(X_model['loan_purpose'])
X_model['insurance_type'] = pd.Categorical(X_model['insurance_type'])
x_train_res.info()
x_train_res['source'] = pd.Categorical(x_train_res['source'])
x_train_res['financial_institution'] = pd.Categorical(x_train_res['financial_institution'])
x_train_res['loan_purpose'] = pd.Categorical(x_train_res['loan_purpose'])
x_train_res['insurance_type'] = pd.Categorical(x_train_res['insurance_type'])
y_model.sum()
X_test.info()
#y_model.to_csv('y_Model.csv')

#y_train_res.value_counts()
'''
def plot_2d_space(X_train, y_train, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X_train[y_train==l, 0],
            X_train[y_train==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
    
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train1 = pca.fit_transform(X_train)
plot_2d_space(X_train1, y_train, 'Imbalanced dataset (2 PCA components)')
pca = PCA(n_components=2)
x_train_res1 = pca.fit_transform(x_train_res)
plot_2d_space(x_train_res1, y_train_res, 'Imbalanced dataset (2 PCA components)')
'''


#XGBOOST

X = X.apply(pd.to_numeric)
y = y.apply(pd.to_numeric)
X_test= X_test.apply(pd.to_numeric)
X_model = X_model.apply(pd.to_numeric)
#y_model = y_model.apply(pd.to_numeric)
df_test = df_test.apply(pd.to_numeric)
x_train_res= x_train_res.apply(pd.to_numeric)





xg_cl = xgb.XGBClassifier(objective='reg:logistic', seed=123)
xg_cl.fit(x_train_res,y_train_res)
predi_xgb = xg_cl.predict(X_test)
xg_cl.score(X_test, y_test)
print(accuracy_score(y_test,predi_xgb))

predi_xgb = pd.Categorical(predi_xgb)
predi_xgb.value_counts()

pred_pro = xg_cl.predict_proba(X_test)
pred_pro = [p[1] for p in pred_pro]
print( roc_auc_score(y_test, pred_pro) )

print(classification_report(y_test, predi_xgb)) 

print(confusion_matrix(y_test, predi_xgb)) 

#final
import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective='reg:logistic', seed=123, subsample=0.7, learning_rate=0.1, n_estimators=500, gamma= 50, colsample_bytree=0.7, max_depth=500, reg_alpha=1, reg_lambda=10)

  xg_cl.fit(X_model,y_model)

#knn.score(X,y)
prediction1 = xg_cl.predict_proba(df_test)
pred_proba = [p[1] for p in prediction1]

df_final2 = pd.DataFrame()
df_final2['loan_id'] = df_test1['loan_id']
df_final2['m13'] = pred_proba
df_final2 = df_final2.sort_values('loan_id')
#df_final2.to_csv('xgb_prob.csv')


#cross validating
churn_matrix = xgb.DMatrix(data=X, label=y)
params= {"objective":"binary:logistic"}
cv_results = xgb.cv(dtrain = churn_matrix, params = params, nfold =6, num_boost_round=100, metrics = "auc", as_pandas= True, seed=123)
print(cv_results)
print((1 - cv_results["test-auc-mean"]).iloc[-1])

#parameter tuning
from sklearn.model_selection import RandomizedSearchCV
gbm_param_grid = {'learning_rate': np.arange(0.1,1.1,0.1), 'n_estimators':[10,50,100,200,500,1000], 'subsample': np.arange(0.05,1.05,0.05)}
gbm_param_grid_1 = {'learning_rate': [0.001,1], 'n_estimators':[100,1000], 'subsample': [0.2,1], 'gamma' : [0.01,10], 'colsample_bytree': [0.3,1], 'max_depth': [50,500]}
gbm_param_grid_2 = {'learning_rate': [1,10], 'n_estimators':[1000,1500], 'subsample': [0.7,1], 'gamma' : [0.1,10], 'colsample_bytree': [0.7,1], 'max_depth': [1000,500]}
gbm_param_grid_3 = {'learning_rate': [1,0.1], 'n_estimators':[500,1000], 'subsample': [0.7], 'gamma' : [1,10,50], 'colsample_bytree': [0.7], 'max_depth': [500], 'reg_alpha':[0.01,1],'reg_lambda':[0.1,10]}

'''
gbm = xgb.XGBClassifier()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions= gbm_param_grid_3, n_iter = 10, scoring='roc_auc', cv=4, verbose=9)
randomized_mse.fit(X, y)
print(randomized_mse.best_params_)
print(randomized_mse.best_score_)
from sklearn.model_selection import GridSearchCV
dmatrix = xgb.DMatrix(data= X, label=y)
gbm_matrix_grid = {'learning_rate': np.arange(0.1,0.3,0.2), 'n_estimators':[25,50,75],'subsample':np.arange(0.35,0.45,0.01) }
gbm_matrix_grid_1 = {'learning_rate': np.arange(0.1,0.3,0.2), 'n_estimators':[75,100,125,150], 'gamma': np.arange(1.5,2.5,0.1), 'max_depth': np.arange(8,10,15)  }
gbm = xgb.XGBClassifier()
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_matrix_grid_1, scoring = 'roc_auc', cv=6, verbose=10)
grid_mse.fit(X, y)
print(grid_mse.best_params_)
print(grid_mse.best_score_)
'''



