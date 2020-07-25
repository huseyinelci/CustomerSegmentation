#===========================    Import Libraies    ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score, GridSearchCV
#from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, fbeta_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, roc_curve, auc)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from category_encoders.woe import WOEEncoder
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
from imblearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl
from collections import Counter
import pickle

#======================    Functions For Cleaning    ==============================
def row_missing_value(df,value=100):
    print('Total ROW numbers Originally...........................:',df.shape[0])
    print('Total ROW numbers After dropping missing rows under {}:'.format(value),
          df[df.isnull().sum(axis=1)<=value].shape[0])
    a = df[df.isnull().sum(axis=1)>value].index
    df.drop(a, inplace=True)

def N_unique(df):
    # The number of Unique values of columns
    N_unique = {}
    for col in df.columns.tolist():
        N_unique[col] = df[col].nunique()
    return N_unique

def report_missing_values(df, value = 27):
    # Missing and Type of columns
    table = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'}) 
    table = table.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    table = table.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0: 'null values (%)'}))
    table = table.T
    table['level'] = table['null values (%)'].apply(lambda x:'Zero' if x==0 else 'High' if x>=value else 'Low' )   
    table['N_unique'] = N_unique(df).values()
    table.sort_values(by=['level','null values (nb)'], ascending=[True,False],inplace=True)
    return table 

def code_to_nan(df, attribute_file):
    '''
    Input:
        df: Missing-Unknown degerleri degisecek dataframe
        df_attribute: Hangi column larda hangi verilere karsilik degisiklik 
                      yapilacaginin bilgisini barindiran dataframe
    Processing:
        1. Standardize the 'missing' and 'unknown' value indicated by df_code to np.nan
        2. Convert a list of value into dictionary with default value np.nan
    Return:
        None 

    -> Create attribute dataframe
    -> Filling 'Attribute' feature with ffill()
    -> Drop null-useless rows
    '''
    df_attribute = pd.read_excel(attribute_file, 
                        header=1, 
                        usecols=['Attribute', 'Meaning', 'Value'])
    df_attribute['Attribute'] = df_attribute['Attribute'].ffill()
    df_attribute.dropna(inplace=True)
    
    # Preparing of Missing-Unknown dict_list from df_attribute
    dict_list = {}
    
    columns= ['Attribute','Meaning','Value']
    mask = df_attribute['Meaning'].str.contains('unknown')
    feat_mask = df_attribute[columns][mask].set_index('Attribute')
    
    for i in feat_mask.index:
        if type(feat_mask['Value'][i])==str:
            flist = [int(x) for x in feat_mask['Value'][i].replace(' ', '').split(',')]
            dict_list[i]=dict.fromkeys(flist, np.nan)
        else:
            dict_list[i]={feat_mask['Value'][i]: np.nan}
    
    # Replace into df    
    for attribute in dict_list.keys():
        try:
            df.loc[:, attribute].replace(dict_list[attribute], inplace=True)
        except:
            continue
            
def encode_map_age(df):
    '''
    
    '''
    df['GEBURTSJAHR'] = df['GEBURTSJAHR'].apply(lambda x: int(x/10))
    map_age = {190:193, 191:193, 192:193, 200:199, 201:np.NaN, 0: np.NaN}
    df['GEBURTSJAHR'].replace(map_age, inplace=True)
    
def encode_df_PJ(df):
    
    '''
    First Column for convert of time series
        {1:0,2:0,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:4,13:4,14:5,15:5}
    
    Other Column FOR two type (Mainstream - Avant Garde)
        {1:0, 2:1, 3:0,  5:0,8:0, 10:0, 12:0, 14:0,
        4:1, 6:1, 7:1, 9:1, 11:1, 13:1, 15:1}
    '''
    
    # PRAEGENDE_JUGENDJAHRE, convert mixed information code into seperate code
    PRAEGENDE_time = {1:0,2:0,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:4,13:4,14:5,15:5}
    PRAEGENDE_avant = {1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:1, 8:0, 9:1, 10:0, 11:1, 12:0, 13:1, 14:0, 15:1}
    
    df['PRAEGENDE_time'] = df['PRAEGENDE_JUGENDJAHRE'].map(PRAEGENDE_time)
    df['PRAEGENDE_avant'] = df['PRAEGENDE_JUGENDJAHRE'].map(PRAEGENDE_avant)
    del df['PRAEGENDE_JUGENDJAHRE']
    
def encode_df_EIN(df):
    '''
        Transform year to index for 'EINGEFUEGT_AM' feature
    '''
    # EINGEFUEGT_AM: transform year to index
    df['EINGEFUEGT_AM'] = pd.to_datetime(df['EINGEFUEGT_AM'])
    
    max_year = df['EINGEFUEGT_AM'].max().year
    df['EINGEFUEGT_ind'] = (max_year - df['EINGEFUEGT_AM'].dt.year)       
    
def encode_df_OWK(df):
    '''
    'OST_WEST_KZ'
    'W': 1
    'O': 0
    '''
    df['OST_WEST_KZ'].replace({'W': 1, 'O': 0}, inplace=True)
    
def drop(df,drop_list):
    df.drop(drop_list, axis=1, inplace=True)

def col_sync(customers,azdias):    
    '''
    Check and sync columns between customers and azdias dataframe
    '''
    excess_col_customer = list(set(customers.columns.tolist()) - set(azdias.columns.tolist()))
    
    return excess_col_customer

def corr_drop(df,val=0.7):
    
    # correlation matrix    
    corr = df.corr().abs()
    upper_limit = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    
    # find features to remove 
    drop_columns = [column for column in upper_limit.columns if any(upper_limit[column] > val)]
    
    # drop fetures
    df.drop(drop_columns, axis=1,inplace=True)
    print('feat # after drop{}'.format(df.shape))
    
#======================     Functions For Attributes     ==========================

def attribute(attribute_file):
    '''
        Create attribute dataframe
        Filling 'Attribute' feature with ffill()
        Drop null-useless rows
    '''
    df_attr = pd.read_excel(attribute_file, 
                        header=1, 
                        usecols=['Attribute', 'Meaning', 'Value'])
    df_attr['Attribute'] = df_attr['Attribute'].ffill()
    df_attr.dropna(inplace=True)
    columns= ['Attribute','Meaning','Value']
    mask = df_attr['Meaning'].str.contains('unknown')
    feat_mask = df_attr[columns][mask].set_index('Attribute')
    
    return feat_mask

def attribute_h2o(attribute_file):
    '''
        Read the information from attributes file
        Create attribute dataframe
        Marked to feature with from numeric to float and  from categoric to enum.
    
    '''
    
    df_attr = pd.read_excel(path_read,
                            header = 1,
                            usecols=['Attribute', 'Meaning'])
    df_attr.dropna(inplace=True)
    df_attr['numeric'] = df_attr['Meaning'].str.startswith('numeric')
    df_attr_dict = (df_attr[['Attribute', 'numeric']]
                    .set_index('Attribute')
                    .to_dict()
                    .get('numeric'))

    feature_dict = {feature: 'float' if numeric else 'category'
                    for feature, numeric in df_attr_dict.items()}
    
    return feature_dict
    
#======================     Functions For Display     ==========================

def printS(model):
    '''
    INPUT:
        model: Model which X and Y scores are requested
    OUTPUT:
        Print bestscore_ and best_param_ after GridSearchCV
    
    Example:
    >>>printS(clf_Train)
    ════════════════════════════════════════════════════════════
    For LGBMClassifier() model
    Best Score: 0.809740807398293
    ════════════════════════════════════════════════════════════
    Best Params: 
     {'class_weight': 'balanced', 'learn_rate': 0.01, 
     ...'max_depth': 8, 'max_features': 25}
    
    '''
    print('═'*60)
    print("For {} model \nBest Score: {}".format(type(model).__name__, model.best_score_))
    print('═'*60)
    print("Best Params: \n", model.best_params_)

def accu_cm(model, X_train, y_train, X_test, y_test):
    '''
    INPUT:
        model  : Model which X and Y scores are requested
        X_test : X-test is the test data set
        y_test : Y-test is the set of labels to all the data in X-test
        
    OUTPUT:
        Print ROC-AUC Score, Accurarcy Score and Confusion Matrix of Model
    
    Example:
    >>>accu_cm(lgb, X_test, y_test)
    ════════════════════════════════════════
    Model Name      : LGBMClassifier
    ════════════════════════════════════════
    ROC-AUC Score   :  0.7453128012338539   calculated with predict_proba
    ROC-AUC Score   :  0.4995757918552036   calculated just predict
    Accurarcy Score :  0.98677962945722
    Confusion Matrix:
     [[10599     9]
     [  133     0]]
    '''
    model = model.fit(X_train, y_train)
    print('═'*60)
    print('Model Name      : {}'.format(type(model).__name__))
    print('═'*60)
    
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print('ROC-AUC Score   : ',score)

    accuracy = accuracy_score(y_test,model.predict(X_test))
    print('Accurarcy Score : ',accuracy)
    
    cm = confusion_matrix(y_test, model.predict(X_test))
    print('Confusion Matrix:\n',cm)

def ROC_AUC_score(model, X_train, y_train, X_test, y_test):
    # Fitting
    model.fit(X_train, y_train)
    
    # Evaluating
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    # Printing 
    print('='*50)
    print("For {} model ROC AUC Score: {}".format(type(model).__name__,score))
    return score

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)    
    
#================     Pipeline Functions For Pre-Processing     ================
    
def pre_processing(df, attribute_file, val, cols=[]):
    '''
    Arg:
        df : Dataframe
        attribute_file : Features information of dataframe
        val : Threshold for Correlation
        cols : Before get_dummies list of df.columns for K-means
    
    Return:
        df : df dataframe after processing 
        df_cols : Before get_dummies list of df.columns
        
    '''
    
    print('Start Preprocessing...')
    # Converting Unknown values to NaN values in df dataframes
    try:
        code_to_nan(df, attribute_file)
    except:
        pass
    # Drop columns of  as high NaN value
    df=df.drop(['D19_LETZTER_KAUF_BRANCHE','EINGEFUEGT_AM'],axis=1)
    
    # Drop columns to avoid lot of columns
    df=df.drop(['ALTER_KIND4', 'ALTER_KIND3', 'ALTER_KIND2', 'ALTER_KIND1'],axis=1)

    # Categorical-Numeric columns adjusting
    categorical_col=df.select_dtypes(exclude='number').columns.tolist()
    numeric_col=df.select_dtypes(include='number').columns.tolist()
    df[numeric_col]=df[numeric_col].apply(pd.to_numeric)
    
    # Decrease of features
    if len(cols)==0:
        # Remove Correlated Features
        print(' Implementing of Corr-Drop')
        corr_drop(df, val=0.8)
    else:
        print(' Implementing of Cols-Drop')
        if 'RESPONSE' in df.columns.tolist():
            drop_list = set(df.columns.tolist()) - set(cols) - set(['RESPONSE'])
        else:
            drop_list = set(df.columns.tolist()) - set(cols)
        df.drop(drop_list, axis=1, inplace=True)

    
    # Encoding
    print(' processing of get_dummies ...')
    df_cols = df.columns.tolist()
    if 'OST_WEST_KZ' in df_cols:
        encode_df_OWK(df)
    
    try:
        df = pd.get_dummies(df, columns=categorical_col, prefix_sep='_', drop_first=True)
    except:
        pass
    # Fillna
    print(' Processing of fill NaN with MEAN...')
    df.fillna(df.mean(), inplace=True)
    
    # Scaling
    print(' Processing of scaling...')
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df, df_cols


#================     Functions For Some Numbers     ================

def calculate_cluster(df, column):
    '''
    The distribution and percentage calculations in the table for entered Column.
    Arg:
        df : Dataframe
        column : The column to be calculated by distribution and percentage.
    
    Return:
        pop : Dataframe with results 
        
    '''
    (pop_unique, pop_counts) = np.unique(df[column], return_counts=True)
    pop_frequencies = np.asarray((pop_unique, pop_counts)).T
    pop = pd.DataFrame(pop_frequencies, columns=['Cluster','Population'])
    pop['Pop_Perc'] = (pop['Population']/pop['Population'].sum()*100).round(3)
    return pop