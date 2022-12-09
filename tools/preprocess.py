import pandas as pd
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import random
import numpy as np
import os
from sklearn.decomposition import PCA
from tools.selector import *




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


def rev_sigmoid(x):
    if x == 5000 :
        return 5000
    return 2/(1+np.exp(0.9*x/1000))

def add_geo(data,places,apply_pca, filepath="data/"):
    X_train_raw=pd.read_csv(filepath +'X_train_J01Z4CN.csv') 
    X_test_raw=pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    X_train_geo=pd.read_pickle(filepath+"geodata/X_train_geodata.pkl")
    X_test_geo=pd.read_pickle(filepath+"geodata/X_test_geodata.pkl")

    for i in X_train_geo.columns :
        if "num" in i :
            X_train_geo[i]= X_train_geo[i].apply(lambda x: rev_sigmoid(x))
            X_test_geo[i]= X_test_geo[i].apply(lambda x: rev_sigmoid(x))
    X_train_geo=X_train_geo.replace([5000], 0)
    X_test_geo=X_test_geo.replace([5000], 0)

    for i in  X_train_geo.columns :
        if not "num" in i and not "rating" in i:
            X_train_geo.drop(inplace=True, columns=i)
            X_test_geo.drop(inplace=True, columns=i)

    # Dropping features with multicolinarity / low importance : 
    columns_to_drope=["num_point_of_interest","num_sublocality_level_1","num_place_of_worship","num_supermarket","num_campground","num_local_government_office",]
    X_train_geo.drop(inplace=True, columns=columns_to_drope)
    X_test_geo.drop(inplace=True,columns=columns_to_drope)
    
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

    if apply_pca:
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(data_geo)
        # fs= FeatureSelector(data = train, labels = target)
        # data_geo = fs.remove(methods = 'all')

    data = data.reset_index(drop=True)
    data = pd.concat([data, data_geo], axis=1)

    return data
    
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

def add_quality(data, filepath="data/"):
    
    print("loaded raw data")
    X_train_raw=pd.read_csv(filepath +'X_train_J01Z4CN.csv') 
    X_test_raw=pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    print("loaded quality image data")
    X_train_quality=pd.read_pickle(filepath+"X_train_quality.pkl")
    X_test_quality=pd.read_pickle(filepath+"X_test_quality.pkl")
    
    print("Converted id to numeric")
    X_train_quality['id_annonce']=pd.to_numeric(X_train_quality['id_annonce'])
    X_test_quality['id_annonce']=pd.to_numeric(X_test_quality['id_annonce'])

    X_train_quality.drop(inplace=True, columns ="image_id")
    X_test_quality.drop(inplace=True, columns ="image_id")

    print(X_train_quality.columns)
    print(X_test_quality.columns)

    print("Grouping min score")
    X_train_quality = X_train_quality.groupby('id_annonce').min()
    X_test_quality = X_test_quality.groupby('id_annonce').min()
     

    print("Reordering indexes")
    # ordering indexes of X_train    
    X_train_quality = X_train_quality.reindex(index=X_train_raw['id_annonce'])

    # Ordering indexes of X_test
    X_test_quality = X_test_quality.reindex(index=X_test_raw['id_annonce'])

    print("Concat Xtest and Xtrain")

    data_images=pd.concat([X_train_quality, X_test_quality], axis=0)

    print("Removing index")

    data_images = data_images.reset_index(drop=True)
    
    data = data.reset_index(drop=True)

    final = pd.concat([data, data_images], axis=1)

    print(final.columns)
    return final

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

def add_polar_rotation(data):
  '''
  # most frequently used degrees are 30,45,60
  input: dataframe containing Latitude(x) and Longitude(y)
  '''
  data["rot_45_x"] = (0.707 * data['approximate_latitude']) + (0.707 * data['approximate_longitude'])
  data["rot_45_y"] = (0.707 * data['approximate_longitude']) + (0.707 * data['approximate_latitude'])
  
  data["rot_30_x"] = (0.866 * data['approximate_latitude']) + (0.5 * data['approximate_longitude'])
  data["rot_30_y"] = (0.866 * data['approximate_longitude']) + (0.5 * data['approximate_latitude'])
  
  return data

def add_polar_coordinates(data):
    data['radius']=np.sqrt((data['approximate_latitude']**2)+(data['approximate_longitude']**2))
    data['angle']=np.arctan2(data['approximate_longitude'],data['approximate_latitude'])
    return data

def add_geo_pca(data):
  '''
  input: dataframe containing Latitude(x) and Longitude(y)
  '''
  coordinates = data[['approximate_latitude','approximate_longitude']].values
  pca_obj = PCA().fit(coordinates)
  pca_x = pca_obj.transform(data[['approximate_latitude', 'approximate_longitude']])[:,0]
  pca_y = pca_obj.transform(data[['approximate_latitude', 'approximate_longitude']])[:,1]

  data["geo_pca_x"]=pca_x
  data["geo_pca_y"]=pca_y
  return data


def preprocess(X_train_0, Y_train_0, X_test_0, parameters):
    """
    Data preprocessing pipeline    
    """

    # Fixing seeds
    print("Fixing seeds")
    seed_everything()
    
    
    # Concatenating data
    print("Concatenated Data")
    data = pd.concat([X_train_0, X_test_0], axis=0).reset_index(drop=True)

    # Dropping columns
    print("Dropping columns")

    data=data.drop(columns=parameters["drop_columns"])

    # Frequency Encoding
    if len(parameters["frequency_encoding"])>0:
        print("Applying frequency encoding")
        for i in parameters["frequency_encoding"]:
            data=frequency_encoder(data,[i])

    
    # Quantile Encoding
    if len(parameters["quantile_encoding"])>0:
        print("Applying Quantile encoding")
        data=quantile_encoder(data,X_train_0, Y_train_0, X_test_0,parameters["quantile_encoding"] )

    
    # Adding polar coordinates
    if parameters["add_polar_coordinates"]:
        print("Adding polar coordinates")
        data=add_polar_coordinates(data)
    
    # Adding polar rotation
    if parameters["add_polar_rotation"]:
        print("Adding polar rotation")
        data=add_polar_rotation(data)

    # adding geo pca data based on lat and long
    if parameters["add_geo_pca"]:
        print("Adding Geo PCA")
        data=add_geo_pca(data)
    


    # impute using images 
    if parameters["images_imputation"]:
        print("Using image feature extraction imputation")
        data= add_images(data, parameters["images_features"])
        data = images_impute(data, True)
    
    # Knn imputation
    if len(parameters["knn_imputation"]) >0 :
        print("Using KNN imputation")
        data = knn_impute_all(data, list_columns=parameters["knn_imputation"])
    

    # Constant imputation
    data.loc[(data['property_type']!="appartement") & data['floor'].isna(), 'floor'] = 0
    #data.loc[ data['exposition'].isna(), 'exposition'] = 0

    data = data.copy()
    # Additional imputation for floor
    if len(parameters["knn_imputation"]) >0 :

        data=knn_impute(data, "floor")

    # Add geodata
    if parameters["add_geo"]:
        data=add_geo(data, parameters["geodata"], parameters["apply_pca_geo"])


    # Adding images
    if parameters["add_images_data"]:
        data=add_images(data, parameters["images_features"])
    

    # Adding images quality
    if parameters["add_quality"]:
        print("Adding image quality")
        data=add_quality(data)

    # Hot encoding
    if parameters["hot_encoding"]:
        print("Hot Encoding")
        data = pd.get_dummies(data)



    # Scaling 
    scaler = StandardScaler()
    rbst_scaler=RobustScaler()
    power_transformer=PowerTransformer()

    if parameters["standard_scaling"]:
        print("Standard scaling")
        data_0=data_2.copy()
        scaler.fit(data)
        data=scaler.transform(data)
        data=pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters["robust_scaling"]:

        print("Robust scaling")

        data_0=data.copy()

        rbst_scaler.fit(data)
        data=rbst_scaler.transform(data)
        data=pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters["power_scaling"]:
        print("Power scaling")

        data_0=data.copy()
        power_transformer.fit(data)
        data=power_transformer.transform(data)
        data=pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # target transformation : 
    if parameters["target_transformation"]:
        Y_train_1=np.log(Y_train_0)["price"]
    else:
        Y_train_1=Y_train_0

    # Splitting the data back
    X_train_1=data.loc[:X_train_0.index.max(),:]
    X_test_1=data.loc[X_train_0.index.max()+1:,:]

    return X_train_1,Y_train_1, X_test_1