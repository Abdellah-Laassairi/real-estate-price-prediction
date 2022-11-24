
import pandas as pd
# Target encoding city
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    # Loading the Data
    # Raw Loaded data
    X_train_raw=pd.read_csv(filepath +'X_train_J01Z4CN.csv') 
    Y_train_raw=pd.read_csv(filepath + 'y_train_OXxrJt1.csv')

    X_test_raw=pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    # Droping ids for training
    X_train_0=X_train_raw.drop(columns="id_annonce")
    Y_train_0=Y_train_raw.drop(columns="id_annonce")

    X_test_0=X_test_raw.drop(columns="id_annonce")

    # Saving Test ids for prediction
    X_test_ids=X_test_raw["id_annonce"]
    X_test_ids.to_pickle("data/X_test_ids.pkl")

    return X_train_0, Y_train_0, X_test_0, X_test_ids



#KNN imputation / Try and expirement different imputations
def knn_impute(df0, column):
    """ 
    """
    # Creating a copy of the input dataframe
    df = df0.copy()

    # numeric_df : subset of df composed only of numerical data type colums
    numeric_df = df.select_dtypes(np.number)

    # full columns : columns that have no missing data
    full_columns=numeric_df.loc[:,numeric_df.isna().sum()==0].columns

    # knn_x_train : training data for the missing values
    knn_x_train = numeric_df.loc[numeric_df[column].isna()==False, full_columns]

    # knn_y_train: target data for the missing valies 
    knn_y_train= numeric_df.loc[numeric_df[column].isna()==False, column]

    # knn_x_test : the data with missing values for the target column
    knn_x_test = numeric_df.loc[numeric_df[column].isna()==True, full_columns]

    # Creating the KNeighbors Regress
    knn=KNeighborsRegressor()

    # Fitting the model
    knn.fit(knn_x_train, knn_y_train)

    y_pred=knn.predict(knn_x_test)

    df.loc[df[column].isna()==True, column]=y_pred

    return df

# Applies knn imputation over a list of columns
def knn_impute_all(df, list_columns):
    """
    """
    for column in list_columns:
        df=knn_impute(df,column)
    return df


def quantile_encoder(df, X_train_0,Y_train_0, X_test_0 , column):
    city_encoder = ce.quantile_encoder.QuantileEncoder()
    df_city_train=city_encoder.fit_transform(X_train_0[column[0]], Y_train_0["price"])
    df_city_train
    df_city_test = city_encoder.transform(X_test_0[column[0]])
    df_city_test
    new_city_column =pd.concat([df_city_train, df_city_test], axis=0).reset_index(drop=True)
    df[column[0]]=new_city_column[column[0]]
    return df

def frequency_encoder(df, column):
    fq = df.groupby(column).size()/len(df)
    # mapping values to dataframe
    df.loc[:, "freq_encode"] = df[column[0]].map(fq)
    # drop original column.
    df = df.drop([column[0]], axis=1)
    df = df.rename(columns={"freq_encode":column[0]})
    data_1=df.copy()
    return data_1

def preprocess(X_train_0, Y_train_0, X_test_0, parameters):
    """
    Data preprocessing pipeline    
    """
    # Concatenating data
    data = pd.concat([X_train_0, X_test_0], axis=0).reset_index(drop=True)

    # Dropping columns
    data_1=data.drop(columns=parameters["drop_columns"])

    # Frequency Encoding
    data_2=frequency_encoder(data_1,parameters["frequency_encoding"] )
    
    # Quantile Encoding
    if len(parameters["quantil_encoding"])>0:
        data_3=quantile_encoder(data_2,X_train_0, Y_train_0, X_test_0,parameters["quantile_encoding"] )
    else:
        data_3 = data_2.copy()

    
    # Knn imputation
    data_4 = knn_impute_all(data_3, list_columns=parameters["knn_imputation"])
    
    # Constant imputation
    data_4.loc[(data_4['property_type']!="appartement") & data_4['floor'].isna(), 'floor'] = 0

    # Additional imputation for floor
    data_5=knn_impute(data_4, "floor")

    # Hot encoding
    data_5 = pd.get_dummies(data_5)

    # Scaling 
    scaler = StandardScaler()
    scaler.fit(data_5)
    data_6=pd.DataFrame(scaler.transform(data_5), index=data_5.index, columns=data_5.columns)

    # target transformation : 
    Y_train_1=np.log(Y_train_0)["price"]

    # Splitting the data back
    X_train_1=data_6.loc[:X_train_0.index.max(),:]
    X_test_1=data_6.loc[X_train_0.index.max()+1:,:]

    return X_train_1,Y_train_1, X_test_1