"""
Helper functions for categorical encodings
"""
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def kfold_target_encoder(train, test,y_train, cols_encode, target, folds=20):
    """
    Mean regularized target encoding based on kfold
    """
    train_new = train.copy()
    train_new = pd.concat([train_new, y_train], axis=1)
    test_new = test.copy()
    kf = KFold(n_splits=folds, random_state=1)
    for col in cols_encode:
        global_mean = train_new[target].mean()
        for train_index, test_index in kf.split(train):
            mean_target = train_new.iloc[train_index].groupby(col)[target].mean()
            train_new.loc[test_index,   "_mean_enc"] = train_new.loc[test_index, col].map(mean_target)
        train_new[ "_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(col)[target].mean()
        test_new["_mean_enc"] = test_new[col].map(col_mean)
        test_new["_mean_enc"].fillna(global_mean, inplace=True)
    
    # filtering only mean enc cols
    train_new = train_new.filter(like="mean_enc", axis=1)
    test_new = test_new.filter(like="mean_enc", axis=1)


    return train_new, test_new
        
def catboost_target_encoder(train, test, cols_encode, target):
    """
    Encoding based on ordering principle
    """
    train_new = train.copy()
    test_new = test.copy()
    for column in cols_encode:
        global_mean = train[target].mean()
        cumulative_sum = train.groupby(column)[target].cumsum() - train[target]
        cumulative_count = train.groupby(column).cumcount()
        train_new[column + "_cat_mean_enc"] = cumulative_sum/cumulative_count
        train_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(column).mean()[column + "_cat_mean_enc"]  #
        test_new[column + "_cat_mean_enc"] = test[column].map(col_mean)
        test_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
    # filtering only mean enc cols
    train_new = train_new.filter(like="cat_mean_enc", axis=1)
    test_new = test_new.filter(like="cat_mean_enc", axis=1)
    return train_new, test_new

