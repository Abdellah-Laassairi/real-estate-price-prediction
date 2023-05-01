import json
import os
import random

import category_encoders as ce
import numpy as np
import pandas as pd
from category_encoders import SummaryEncoder
from category_encoders import WOEEncoder
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from tools.encoders import *
from tools.selector import *


def seed_everything(seed=42):
    """ "
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


import pandas as pd
import fuzzywuzzy.process as fwp
from unidecode import unidecode


def standards2(row):
    row['Nom Commune'] = str(row['Nom Commune'])
    result = unidecode(row['Nom Commune'].lower())
    return result


def standards(row):
    result = unidecode(row.city.lower())
    return result


fr = pd.read_csv('data/fr.csv')
fr['city'] = fr.apply(standards, axis=1)

fr2 = pd.read_excel(
    'data/Niveau_de_vie_2013_a_la_commune-Global_Map_Solution.xlsx')
fr2['Nom Commune'] = fr2.apply(standards2, axis=1)

with open('data/city_names.json') as f:
    database = json.load(f)


def fmatch(row):
    return database[row.city]


def capital(row):
    if len(fr.loc[fr['city'] == row.df2_name, 'capital'].values) < 1:
        return None
    else:
        return fr.loc[fr['city'] == row.df2_name, 'capital'].values[0]


def population(row):
    if len(fr.loc[fr['city'] == row.df2_name, 'population'].values) < 1:
        return None
    else:
        return fr.loc[fr['city'] == row.df2_name, 'population'].values[0]


def nvc(row):
    if (len(fr2.loc[fr2['Nom Commune'] == row.df3_name,
                    'Niveau de vie Commune'].values) < 1):
        return None
    else:
        return fr2.loc[fr2['Nom Commune'] == row.df3_name,
                       'Niveau de vie Commune'].values[0]


def nvd(row):
    if (len(fr2.loc[fr2['Nom Commune'] == row.df3_name,
                    'Niveau de vie Département'].values) < 1):
        return None
    else:
        return fr2.loc[fr2['Nom Commune'] == row.df3_name,
                       'Niveau de vie Département'].values[0]


def population_proper(row):
    if len(fr.loc[fr['city'] == row.df2_name, 'population_proper'].values) < 1:
        return None
    else:
        return fr.loc[fr['city'] == row.df2_name,
                      'population_proper'].values[0]


def lng(row):
    if len(fr.loc[fr['city'] == row.df2_name, 'lng'].values) < 1:
        return None
    else:
        return fr.loc[fr['city'] == row.df2_name, 'lng'].values[0]


def lat(row):
    if len(fr.loc[fr['city'] == row.df2_name, 'lat'].values) < 1:
        return None
    else:
        return fr.loc[fr['city'] == row.df2_name, 'lat'].values[0]


def load_data(filepath, add_geodata=False):
    # Loading the Data

    # Raw Loaded data
    X_train_raw = pd.read_csv(filepath + 'X_train_J01Z4CN.csv')
    Y_train_raw = pd.read_csv(filepath + 'y_train_OXxrJt1.csv')

    X_test_raw = pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    # Droping ids for training
    X_train_0 = X_train_raw.drop(columns='id_annonce')
    Y_train_0 = Y_train_raw.drop(columns='id_annonce')

    X_test_0 = X_test_raw.drop(columns='id_annonce')

    X_train_0['city'] = X_train_0.apply(standards, axis=1)
    X_test_0['city'] = X_test_0.apply(standards, axis=1)

    # Saving Test ids for prediction
    X_test_ids = X_test_raw['id_annonce']
    X_test_ids.to_pickle('data/X_test_ids.pkl')

    return X_train_0, Y_train_0, X_test_0, X_test_ids


def load_hyperparameters(path='models/hyperparameters.json'):
    with open(path) as f:
        data = json.load(f)
    xgb_params = data['xgb_params']
    lgb_params = data['lgb_params']
    cat_params = data['cat_params']
    return xgb_params, lgb_params, cat_params


def rev_sigmoid(x):
    if x == 5000:
        return 5000
    return 2 / (1 + np.exp(0.9 * x / 1000))


def add_geo(data, places, filepath='data/'):
    X_train_raw = pd.read_csv(filepath + 'X_train_J01Z4CN.csv')
    X_test_raw = pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    X_train_geo = pd.read_pickle(filepath + 'geodata/X_train_geodata.pkl')
    X_test_geo = pd.read_pickle(filepath + 'geodata/X_test_geodata.pkl')

    for i in X_train_geo.columns:
        if 'num' in i:
            X_train_geo[i] = X_train_geo[i].apply(lambda x: rev_sigmoid(x))
            X_test_geo[i] = X_test_geo[i].apply(lambda x: rev_sigmoid(x))
    X_train_geo = X_train_geo.replace([5000], 0)
    X_test_geo = X_test_geo.replace([5000], 0)

    for i in X_train_geo.columns:
        if not 'num' in i and not 'rating' in i:
            X_train_geo.drop(inplace=True, columns=i)
            X_test_geo.drop(inplace=True, columns=i)

    # Dropping features with multicolinarity / low importance :
    columns_to_drope = [
        'num_point_of_interest',
        'num_sublocality_level_1',
        'num_place_of_worship',
        'num_supermarket',
        'num_campground',
        'num_local_government_office',
    ]
    X_train_geo.drop(inplace=True, columns=columns_to_drope)
    X_test_geo.drop(inplace=True, columns=columns_to_drope)

    # ordering indexes of X_train
    X_train_geo = X_train_geo.reset_index()
    if len(places) > 0:
        X_train_geo = X_train_geo[places]

    X_train_geo = X_train_geo.set_index('index')
    X_train_geo = X_train_geo.reindex(index=X_train_raw['id_annonce'])
    X_train_geo = X_train_geo.reset_index()

    # Ordering indexes of X_test
    X_test_geo = X_test_geo.reset_index()
    if len(places) > 0:
        X_test_geo = X_test_geo[places]

    X_test_geo = X_test_geo.set_index('index')
    X_test_geo = X_test_geo.reindex(index=X_test_raw['id_annonce'])
    X_test_geo = X_test_geo.reset_index()

    data_geo = pd.concat([X_train_geo, X_test_geo], axis=0)
    data_geo = data_geo.reset_index(drop=True)
    data_geo = data_geo.drop(['id_annonce'], axis=1)

    data = data.reset_index(drop=True)
    data = pd.concat([data, data_geo], axis=1)

    return data


def add_classification_quality(data,
                               features,
                               threshold=0.5,
                               filepath='data/'):
    X_train_raw = pd.read_csv(filepath + 'X_train_J01Z4CN.csv')
    X_test_raw = pd.read_csv(filepath + 'X_test_BEhvxAN.csv')

    X_train_images = pd.read_pickle(
        filepath + 'classification_quality/X_train_images_final.pkl')
    X_test_images = pd.read_pickle(
        filepath + 'classification_quality/X_test_images_final.pkl')

    # Threshhold for predictions
    X_train_images.loc[X_train_images['SCORE_LABEL'] < threshold,
                       'PREDICTED_LABEL'] = 'OTHER'
    X_test_images.loc[X_test_images['SCORE_LABEL'] < threshold,
                      'PREDICTED_LABEL'] = 'OTHER'

    # Aggregating results
    X_train_images = (X_train_images.groupby(['id_annonce',
                                              'PREDICTED_LABEL']).agg({
                                                  'image_quality':
                                                  ['mean', 'sum', 'count']
                                              }).unstack().reset_index())
    X_test_images = (X_test_images.groupby(['id_annonce',
                                            'PREDICTED_LABEL']).agg({
                                                'image_quality':
                                                ['mean', 'sum', 'count']
                                            }).unstack().reset_index())

    # Renaming and flattening index
    X_train_images.columns = [
        '_'.join(a) for a in X_train_images.columns.to_flat_index()
    ]
    X_train_images.rename(columns={'id_annonce__': 'id_annonce'}, inplace=True)

    X_test_images.columns = [
        '_'.join(a) for a in X_test_images.columns.to_flat_index()
    ]
    X_test_images.rename(columns={'id_annonce__': 'id_annonce'}, inplace=True)

    X_train_images['id_annonce'] = pd.to_numeric(X_train_images['id_annonce'])
    X_test_images['id_annonce'] = pd.to_numeric(X_test_images['id_annonce'])

    # ordering indexes of X_train
    X_train_images.set_index('id_annonce', inplace=True)
    if len(features) > 0:
        X_train_images = X_train_images[features]

    X_train_images = X_train_images.reindex(index=X_train_raw['id_annonce'])

    # Ordering indexes of X_test
    X_test_images.set_index('id_annonce', inplace=True)
    if len(features) > 0:
        X_test_images = X_test_images[features]

    X_test_images = X_test_images.reindex(index=X_test_raw['id_annonce'])

    data_images = pd.concat([X_train_images, X_test_images], axis=0)

    data_images = data_images.reset_index(drop=True)

    # data_images=data_images.drop(["id_annonce"])

    data = data.reset_index(drop=True)

    return pd.concat([data, data_images], axis=1)


def quantile_encoder(df, X_train_0, Y_train_0, X_test_0, column):
    city_encoder = ce.quantile_encoder.SummaryEncoder(
        quantiles=[0.25, 0.5, 0.75], m=90000.0)

    cv = KFold(n_splits=5)
    folds = cv.split(X=X_train_0[column[0]])

    for train_idx, test_idx in folds:
        X_train = np.asarray(X_train_0[column[0]])[train_idx]
        X_test = np.asarray(X_train_0[column[0]])[test_idx]
        y_train = np.asarray(Y_train_0['price'])[train_idx]
        y_test = np.asarray(Y_train_0['price'])[test_idx]

        city_encoder.fit(X_train, y_train)

    df_city_train = city_encoder.transform(np.asarray(X_train_0[column[0]]))
    df_city_test = city_encoder.transform(np.asarray(X_test_0[column[0]]))

    new_city_column = pd.concat([df_city_train, df_city_test],
                                axis=0).reset_index(drop=True)

    for column in new_city_column.columns:
        df[column] = new_city_column[column]

    df.drop('city', axis=1, inplace=True)
    return df


def label_encoder(data, column):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return data


def frequency_encoder(df, column):
    L = 8643
    fq = df.groupby(column).size() / L
    # mapping values to dataframe
    df.loc[:, 'freq_encode'] = df[column[0]].map(fq)

    # df.loc[df["freq_encode"]<0.001, "city"] = "other"
    # df.drop("freq_encode", inplace=True, axis=1)
    # drop original column.
    df = df.drop([column[0]], axis=1)
    df = df.rename(columns={'freq_encode': column[0]})

    return df


def add_polar_rotation(data, geo_population):
    """
    # most frequently used degrees are 30,45
    input: dataframe containing Latitude(x) and Longitude(y)
    """
    data['rot_45_x'] = (0.707 * data['approximate_latitude']) + (
        0.707 * data['approximate_longitude'])
    data['rot_45_y'] = (0.707 * data['approximate_longitude']) + (
        0.707 * data['approximate_latitude'])

    data['rot_30_x'] = (0.866 * data['approximate_latitude']) + (
        0.5 * data['approximate_longitude'])
    data['rot_30_y'] = (0.866 * data['approximate_longitude']) + (
        0.5 * data['approximate_latitude'])

    if geo_population:
        data['rot_45_x_city'] = (0.707 * data['lat']) + (0.707 * data['lng'])
        data['rot_45_y_city'] = (0.707 * data['lng']) + (0.707 * data['lat'])

        data['rot_30_x_city'] = (0.866 * data['lat']) + (0.5 * data['lng'])
        data['rot_30_y_city'] = (0.866 * data['lng']) + (0.5 * data['lat'])

    return data


def haversine_dist(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    radius = 6371  # Earth's radius taken from google
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2)**2
    h = 2 * radius * np.arcsin(np.sqrt(d))
    return h


def add_distance_to_center(data):
    data['new_distance'] = data.apply(
        lambda row: haversine_dist(
            row['approximate_latitude'],
            row['approximate_longitude'],
            row['lat'],
            row['lng'],
        ),
        axis=1,
    )
    return data


def add_geopopulation_2(data):
    data['df3_name'] = data.apply(fmatch, axis=1)
    data['nvc'] = data.apply(nvc, axis=1)
    data['nvd'] = data.apply(nvd, axis=1)

    data.drop(columns=['df3_name'], inplace=True)
    return data


def add_geopopulation(data):
    data['df2_name'] = data.apply(fmatch, axis=1)
    data['lng'] = data.apply(lng, axis=1)
    data['lat'] = data.apply(lat, axis=1)

    data['capital'] = data.apply(capital, axis=1)

    data['population'] = data.apply(population, axis=1)

    data['population_proper'] = data.apply(population_proper, axis=1)

    data.drop(columns=['df3_name'], inplace=True)
    return data


def add_polar_coordinates(data, geo_population):
    data['radius'] = np.sqrt((data['approximate_latitude']**2) +
                             (data['approximate_longitude']**2))
    data['angle'] = np.arctan2(data['approximate_longitude'],
                               data['approximate_latitude'])
    if geo_population:
        data['radius_city'] = np.sqrt((data['lat']**2) + (data['lng']**2))
        data['angle_city'] = np.arctan2(data['lng'], data['lat'])
    return data


def add_geo_pca(data, geo_population):
    """
    input: dataframe containing Latitude(x) and Longitude(y)
    """
    coordinates = data[['approximate_latitude',
                        'approximate_longitude']].values
    pca_obj = PCA().fit(coordinates)
    pca_x = pca_obj.transform(
        data[['approximate_latitude', 'approximate_longitude']])[:, 0]
    pca_y = pca_obj.transform(
        data[['approximate_latitude', 'approximate_longitude']])[:, 1]

    data['geo_pca_x'] = pca_x
    data['geo_pca_y'] = pca_y

    if geo_population:
        print('fix nan later')
        # coordinates = data[['lat','lng']].values
        # pca_obj = PCA().fit(coordinates)
        # pca_x = pca_obj.transform(data[['lat', 'lng']])[:,0]
        # pca_y = pca_obj.transform(data[['lat', 'lng']])[:,1]

        # data["geo_pca_x_city"]=pca_x
        # data["geo_pca_y_city"]=pca_y
    return data


def preprocess(X_train_0, Y_train_0, X_test_0, parameters):
    """
    Data preprocessing pipeline
    """
    # Fixing seeds
    seed_everything()

    # target transformation :
    if parameters['target_transformation']:
        Y_train_1 = np.log(Y_train_0)['price']
    else:
        Y_train_1 = Y_train_0

    # #Kfol target encoding
    if len(parameters['target_encoding']) > 0:
        X_train_enc, X_test_enc = kfold_target_encoder(
            X_train_0,
            X_test_0,
            Y_train_0,
            parameters['target_encoding'],
            'price',
            folds=10,
        )
        X_train_0[parameters['target_encoding'][0]] = X_train_enc['_mean_enc']
        X_test_0[parameters['target_encoding'][0]] = X_test_enc['_mean_enc']

    # Concatenating data
    data = pd.concat([X_train_0, X_test_0], axis=0).reset_index(drop=True)

    # data['city']= data['city']+data['property_type']

    # Adding geopopulation data
    if parameters['add_geopopulation']:
        data = add_geopopulation(data)

    if parameters['add_geopopulation_2']:
        data = add_geopopulation_2(data)

    # Dropping columns
    data = data.drop(columns=parameters['drop_columns'])

    # Add distance to city center
    if parameters['add_distance_to_city_center']:
        data = add_distance_to_center(data)

    # Frequency Encoding
    if len(parameters['frequency_encoding']) > 0:
        for i in parameters['frequency_encoding']:
            data[i] = data[i].apply(str)
            data = frequency_encoder(data, [i])

    # Label Encoding
    if len(parameters['label_encoding']) > 0:
        for i in parameters['label_encoding']:
            data[i] = data[i].apply(str)
            data = label_encoder(data, i)

    # Quantile Encoding
    if len(parameters['quantile_encoding']) > 0:
        data = quantile_encoder(data, X_train_0, Y_train_0, X_test_0,
                                parameters['quantile_encoding'])

    # Adding polar coordinates
    if parameters['add_polar_coordinates']:
        data = add_polar_coordinates(data, parameters['add_geopopulation'])

    # Adding polar rotation
    if parameters['add_polar_rotation']:
        data = add_polar_rotation(data, parameters['add_geopopulation'])

    # adding geo pca data based on lat and long
    if parameters['add_geo_pca']:
        data = add_geo_pca(data, parameters['add_geopopulation'])

    # Constant imputation floor
    if parameters['constant_imputation_floor']:
        data.loc[(data['property_type'] != 'appartement')
                 & (data['property_type'] != 'chambre')
                 & data['floor'].isna(), 'floor', ] = 0

    # Constant imputation land size
    if parameters['constant_land_size']:
        data.loc[(data['property_type'] == 'chambre')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'duplex') &
                 (data['land_size'].isna()), 'land_size', ] = 0

    # constant imputation energy value
    if parameters['constant_energy_performance_value']:
        data.loc[(data['property_type'] == 'terrain')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'hôtel particulier')
                 & (data['energy_performance_value'].isna()),
                 'energy_performance_value', ] = 0

    # constant imputation ghg value
    if parameters['constant_ghg_value']:
        data.loc[(data['property_type'] == 'terrain à bâtir')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'hôtel particulier')
                 & (data['ghg_value'].isna()), 'ghg_value', ] = 0

    # constant imputation bedrooms
    if parameters['constant_imputation_bedrooms']:
        data.loc[(data['property_type'] == 'chambre') &
                 (data['nb_bedrooms'].isna()), 'nb_bedrooms', ] = 0

    # Constant imputation exposition
    if parameters['constant_imputation_exposition']:
        data.loc[(data['property_type'] == 'péniche')
                 | (data['property_type'] == 'propriété')
                 | (data['property_type'] == 'parking')
                 | (data['property_type'] == 'terrain')
                 | (data['property_type'] == 'terrain à bâtir')
                 | (data['property_type'] == 'viager')
                 | (data['property_type'] == 'divers') &
                 (data['exposition'].isna()), 'exposition', ] = 0

    # Add geodata
    if parameters['add_geo']:
        data = add_geo(data, parameters['geodata'])

    # Adding images classification and quality
    if parameters['add_classification_quality']:
        data = add_classification_quality(
            data, parameters['images_features'],
            parameters['classification_threshold'])

    # Hot encoding
    if parameters['hot_encoding']:
        data = pd.get_dummies(data)

    # Mean Imputation
    if parameters['mean_imputation']:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_0 = data.copy()
        imp_mean.fit(data)
        data = imp_mean.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # Minimum imputation
    if parameters['mini_imputation']:
        na_columns = [
            'size',
            'floor',
            'land_size',
            'energy_performance_value',
            'ghg_value',
            'nb_rooms',
            'nb_bedrooms',
            'nb_bathrooms',
        ]
        for column in na_columns:
            data[column].fillna(value=data[column].min(), inplace=True)

    # Iter Imputation
    if parameters['iter_imputation']:
        iter_mean = IterativeImputer(estimator=LGBMRegressor(), random_state=0)
        data_0 = data.copy()
        iter_mean.fit(data)
        data = iter_mean.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # Feautures interractions
    if parameters['features_interactions']:
        data_0 = data.copy()
        poly = PolynomialFeatures(2)
        data = poly.fit_transform(data)
        data = pd.DataFrame(data,
                            index=data_0.index,
                            columns=poly.get_feature_names(data_0.columns))

    # Scaling
    scaler = StandardScaler()
    rbst_scaler = RobustScaler()
    power_transformer = PowerTransformer()

    if parameters['standard_scaling']:
        data_0 = data.copy()
        scaler.fit(data)
        data = scaler.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters['robust_scaling']:
        data_0 = data.copy()
        rbst_scaler.fit(data)
        data = rbst_scaler.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters['power_scaling']:
        data_0 = data.copy()
        power_transformer.fit(data)
        data = power_transformer.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # Splitting the data back
    X_train_1 = data.loc[:X_train_0.index.max(), :]
    X_test_1 = data.loc[X_train_0.index.max() + 1:, :]

    return X_train_1, Y_train_1, X_test_1
