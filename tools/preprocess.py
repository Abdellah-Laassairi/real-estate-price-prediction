import pandas as pd
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import random
import numpy as np
import os

def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_data(filepath, add_geodata=False):
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

def add_geo(data,places, filepath="data/"):
    X_train_raw=pd.read_csv(filepath +'X_train_J01Z4CN.csv') 
    X_test_raw=pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    #places = ["id_annonce", "hospital"]
    X_train_geo=pd.read_pickle(filepath+"geodata/X_train_geodata.pkl")
    X_test_geo=pd.read_pickle(filepath+"geodata/X_test_geodata.pkl")
    
    # ordering indexes of X_train
    X_train_geo = X_train_geo.reset_index()
    if len(places)>0:
        X_train_geo = X_train_geo[places]
    
    X_train_geo = X_train_geo.set_index('index')
    X_train_geo = X_train_geo.reindex(index=X_train_raw['id_annonce'])
    X_train_geo = X_train_geo.reset_index()

    # Ordering indexes of X_test
    X_test_geo = X_test_geo.reset_index()
    if len(places)>0:
        X_test_geo = X_test_geo[places]


    X_test_geo = X_test_geo.set_index('index')
    X_test_geo = X_test_geo.reindex(index=X_test_raw['id_annonce'])
    X_test_geo = X_test_geo.reset_index()

    data_geo=pd.concat([X_train_geo, X_test_geo], axis=0)
    data_geo = data_geo.reset_index(drop=True)
    data_geo=data_geo.drop(["id_annonce"], axis=1)
    data = data.reset_index(drop=True)
    return pd.concat([data, data_geo], axis=1)


def add_images(data,features, filepath="data/"):
    X_train_raw=pd.read_csv(filepath +'X_train_J01Z4CN.csv') 
    X_test_raw=pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    X_train_images=pd.read_pickle(filepath+"images/X_train_images.pkl")
    X_test_images=pd.read_pickle(filepath+"images/X_test_images.pkl")
    
    X_train_images['id_annonce']=pd.to_numeric(X_train_images['id_annonce'])

    X_test_images['id_annonce']=pd.to_numeric(X_test_images['id_annonce'])

    # ordering indexes of X_train
    X_train_images.set_index("id_annonce", inplace=True)
    if len(features)>0:
        X_train_images = X_train_images[features]
    
    X_train_images = X_train_images.reindex(index=X_train_raw['id_annonce'])

    # Ordering indexes of X_test
    X_test_images.set_index("id_annonce", inplace=True)
    if len(features)>0:
        X_test_images = X_test_images[features]
        
    X_test_images = X_test_images.reindex(index=X_test_raw['id_annonce'])

    data_images=pd.concat([X_train_images, X_test_images], axis=0)

    data_images = data_images.reset_index(drop=True)
    #data_images=data_images.drop(["id_annonce"])
    
    data = data.reset_index(drop=True)

    return pd.concat([data, data_images], axis=1)

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

    # knn_y_train: target data for the missing values 
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

def images_impute(data_2, drop_images=True):

    y=data_2.loc[data_2["nb_bedrooms"].isna()==True, ["n_Bedroom"]].squeeze()
    data_2.loc[data_2["nb_bedrooms"].isna()==True, ["nb_bedrooms"]] =y

    y=data_2.loc[data_2["nb_bathrooms"].isna()==True, ["n_Bathroom"]].squeeze()
    data_2.loc[data_2["nb_bathrooms"].isna()==True, ["nb_bathrooms"]] =y

    # y=data_2.loc[data_2["nb_rooms"].isna()==True, ["n_livingRoom"]].squeeze()
    # data_2.loc[data_2["nb_rooms"].isna()==True, ["nb_rooms"]] =y

    if(drop_images==True):
        data_2.drop(["n_Bedroom", "n_Bathroom","n_Backyard","n_Frontyard","n_Kitchen","n_livingRoom" ], inplace=True, axis=1)
    return data_2

def quantile_encoder(df, X_train_0,Y_train_0, X_test_0 , column):
    city_encoder = ce.quantile_encoder.QuantileEncoder(quantile=0.5, m=1.0)

    df_city_train=city_encoder.fit_transform(X_train_0[column[0]], Y_train_0["price"])
    df_city_test = city_encoder.transform(X_test_0[column[0]])

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

    # Fixing seeds
    seed_everything()
    
    
    # Concatenating data
    data = pd.concat([X_train_0, X_test_0], axis=0).reset_index(drop=True)

    # Dropping columns
    data_1=data.drop(columns=parameters["drop_columns"])

    # Frequency Encoding
    if len(parameters["frequency_encoding"])>0:
        data_2=frequency_encoder(data_1,parameters["frequency_encoding"] )
    else:
        data_2 = data_1.copy() 
    
    # Quantile Encoding
    if len(parameters["quantile_encoding"])>0:
        data_3=quantile_encoder(data_2,X_train_0, Y_train_0, X_test_0,parameters["quantile_encoding"] )
    else:
        data_3 = data_2.copy()


    # impute using images 
    if parameters["images_imputation"]:
        data_3= add_images(data_3, parameters["images_features"])
        data_3 = images_impute(data_3, True)
    
    # Knn imputation
    data_4 = knn_impute_all(data_3, list_columns=parameters["knn_imputation"])
    
    # Constant imputation
    data_4.loc[(data_4['property_type']!="appartement") & data_4['floor'].isna(), 'floor'] = 0

    # Additional imputation for floor
    data_5=knn_impute(data_4, "floor")

    # Add geodata
    if parameters["add_geo"]:
        data_5=add_geo(data_5, parameters["geodata"])
    
    # Adding images
    if parameters["add_images_data"]:
        data_5=add_images(data_5, parameters["images_features"])

    # Hot encoding
    data_5 = pd.get_dummies(data_5)

    # Scaling 
    scaler = StandardScaler()
    rbst_scaler=RobustScaler()
    power_transformer=PowerTransformer()

    if parameters["standard_scaling"]:
        scaler.fit(data_5)
        data_s=scaler.transform(data_5)

    if parameters["robust_scaling"]:
        rbst_scaler.fit(data_5)
        data_s=rbst_scaler.transform(data_5)

    if parameters["power_scaling"]:
        power_transformer.fit(data_5)
        data_s=power_transformer.transform(data_5)
    
    data_6=pd.DataFrame(data_s, index=data_5.index, columns=data_5.columns)

    # target transformation : 
    Y_train_1=np.log(Y_train_0)["price"]

    # Splitting the data back
    X_train_1=data_6.loc[:X_train_0.index.max(),:]
    X_test_1=data_6.loc[X_train_0.index.max()+1:,:]

    return X_train_1,Y_train_1, X_test_1