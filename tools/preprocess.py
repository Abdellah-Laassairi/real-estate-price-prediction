import json
import os
import random

import category_encoders as ce
import fuzzywuzzy.process as fwp
import numpy as np
import pandas as pd
import yaml
from category_encoders import SummaryEncoder
from category_encoders import WOEEncoder
from lightgbm import LGBMRegressor
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from unidecode import unidecode
from yaml.loader import SafeLoader

from tools.encoders import *
from tools.lema import *
from tools.selector import *

FR_PATH = 'data/other/fr.csv'
NV_PATH = 'data/other/Niveau_de_vie_2013_a_la_commune-Global_Map_Solution.xlsx'
CITIES_PATH = 'data/other/city_names.json'

console = Console()


def standards(row):
    result = unidecode(row.city.lower())
    return result


def standards2(row):
    row['Nom Commune'] = str(row['Nom Commune'])
    result = unidecode(row['Nom Commune'].lower())
    return result


FR = pd.read_csv(FR_PATH)
FR['city'] = FR.apply(standards, axis=1)

NV = pd.read_excel(NV_PATH)
NV['Nom Commune'] = NV.apply(standards2, axis=1)


def seed_everything(seed=42):
    """
    Seed everything.
    """
    console.log('Fixing seeds ')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


with open(CITIES_PATH) as f:
    database = json.load(f)


def fmatch(row):
    return database[row.city]


def capital(row):
    if len(FR.loc[FR['city'] == row.df2_name, 'capital'].values) < 1:
        return None
    else:
        return FR.loc[FR['city'] == row.df2_name, 'capital'].values[0]


def population(row):
    if len(FR.loc[FR['city'] == row.df2_name, 'population'].values) < 1:
        return None
    else:
        return FR.loc[FR['city'] == row.df2_name, 'population'].values[0]


def nvc(row):
    if (len(NV.loc[NV['Nom Commune'] == row.df3_name,
                   'Niveau de vie Commune'].values) < 1):
        return None
    else:
        return NV.loc[NV['Nom Commune'] == row.df3_name,
                      'Niveau de vie Commune'].values[0]


def nvd(row):
    if (len(NV.loc[NV['Nom Commune'] == row.df3_name,
                   'Niveau de vie Département'].values) < 1):
        return None
    else:
        return NV.loc[NV['Nom Commune'] == row.df3_name,
                      'Niveau de vie Département'].values[0]


def population_proper(row):
    if len(FR.loc[FR['city'] == row.df2_name, 'population_proper'].values) < 1:
        return None
    else:
        return FR.loc[FR['city'] == row.df2_name,
                      'population_proper'].values[0]


def lng(row):
    if len(FR.loc[FR['city'] == row.df2_name, 'lng'].values) < 1:
        return None
    else:
        return FR.loc[FR['city'] == row.df2_name, 'lng'].values[0]


def lat(row):
    if len(FR.loc[FR['city'] == row.df2_name, 'lat'].values) < 1:
        return None
    else:
        return FR.loc[FR['city'] == row.df2_name, 'lat'].values[0]


def load_data(filepath, drop_ids=True):
    # Loading the Data

    # Raw Loaded data
    X_train_raw = pd.read_csv(filepath + 'X_train_J01Z4CN.csv')
    Y_train_raw = pd.read_csv(filepath + 'y_train_OXxrJt1.csv')

    X_test_raw = pd.read_csv(filepath + 'X_test_BEhvxAN.csv')
    X_train_ids = X_train_raw['id_annonce']
    # Droping ids for training
    if drop_ids:
        X_train_0 = X_train_raw.drop(columns='id_annonce')
        Y_train_0 = Y_train_raw.drop(columns='id_annonce')

    else:
        X_train_0 = X_train_raw
        Y_train_0 = Y_train_raw

    X_test_0 = X_test_raw.drop(columns='id_annonce')

    X_train_0['city'] = X_train_0.apply(standards, axis=1)
    X_test_0['city'] = X_test_0.apply(standards, axis=1)

    # Saving Test ids for prediction
    X_test_ids = X_test_raw['id_annonce']
    X_test_ids.to_pickle('data/cache/X_test_ids.pkl')

    return X_train_0, Y_train_0, X_test_0, X_test_ids, X_train_ids


def load_hyperparameters(path='models/hyperparameters.yaml'):
    console.log('[bold green]Loading hyperparameters...')

    with open(path, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    xgb_params = data['xgb_params']
    lgb_params = data['lgb_params']
    cat_params = data['cat_params']
    return xgb_params, lgb_params, cat_params


def rev_sigmoid(x):
    if x == 5000:
        return 5000
    return 2 / (1 + np.exp(0.9 * x / 1000))


def add_geo(data, places, filepath='data/'):
    console.log('Adding geodata ')

    X_train_raw = pd.read_csv(filepath + 'tabular/X_train_J01Z4CN.csv')
    X_test_raw = pd.read_csv(filepath + 'tabular/X_test_BEhvxAN.csv')

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

    console.log('Adding classifcation quality using NIMA')
    X_train_raw = pd.read_csv(filepath + 'tabular/X_train_J01Z4CN.csv')
    X_test_raw = pd.read_csv(filepath + 'tabular/X_test_BEhvxAN.csv')

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

    data = data.reset_index(drop=True)

    return pd.concat([data, data_images], axis=1)


def quantile_encoder(df, X_train_0, Y_train_0, X_test_0, column):
    console.log(f'Quantile Encoding : {column}')
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
    console.log(f'Label encoding {column}')
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    return data


def frequency_encoder(df, col):
    console.log(f'Frequency encoding {col}')
    L = len(df[col[0]].unique())
    fq = df.groupby(col).size() / L
    # mapping values to dataframe
    df.loc[:, 'freq_encode'] = df[col[0]].map(fq)

    df = df.drop([col[0]], axis=1)
    df = df.rename({'freq_encode': col[0]})
    return df


def add_polar_rotation(data, angles, geo_population):
    """
    # most frequently used degrees are 30,45
    input: dataframe containing Latitude(x) and Longitude(y)
    """
    console.log(f'Adding polar angles : {angles}')
    x = data['approximate_latitude']
    y = data['approximate_longitude']

    for angle in angles:
        cos_angle = np.cos(angle * np.pi / 180)
        sin_angle = np.sin(angle * np.pi / 180)

        data[f'rot_{angle}_x'] = (cos_angle * x) - (sin_angle * y)
        data[f'rot_{angle}_y'] = (sin_angle * x) + (cos_angle * y)

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
    console.log('Adding distance to city center')
    data['new_distance'] = data.apply(
        lambda row: haversine_dist(
            row['approximate_latitude'],
            row['approximate_longitude'],
            row['lat'],
            row['lng'],
        ),
        axis=1,
    )

    console.log('Adding PCA to distance to city center')
    coordinates = data[['new_distance']].values
    imp = SimpleImputer(strategy='mean')
    coordinates = imp.fit_transform(coordinates)
    pca_obj = PCA().fit_transform(coordinates)
    data['distance_pca'] = pca_obj
    return data


def add_geopopulation_2(data):

    console.log('Adding geopopulation 2')

    try:
        geopopulation = pd.read_csv('data/geodata/geodata_2.csv')
        data = pd.concat([data, geopopulation], axis=1)
    except Exception as e:
        geopopulation = pd.DataFrame({})

        data['df3_name'] = data.apply(fmatch, axis=1)
        data['nvc'] = data.apply(nvc, axis=1)
        data['nvd'] = data.apply(nvd, axis=1)

        data.drop(columns=['df3_name'], inplace=True)
        geopopulation = data[['nvc', 'nvd']]
        geopopulation.to_csv('data/geodata/geodata_2.csv', index=False)

    return data


def add_geopopulation(data):

    console.log('Adding geopopulation 1')

    try:
        geopopulation = pd.read_csv('data/geodata/geodata_1.csv')
        data = pd.concat([data, geopopulation], axis=1)

    except Exception as e:
        geopopulation = pd.DataFrame({})
        data['df2_name'] = data.apply(fmatch, axis=1)
        data['lng'] = data.apply(lng, axis=1)
        data['lat'] = data.apply(lat, axis=1)

        data['capital'] = data.apply(capital, axis=1)

        data['population'] = data.apply(population, axis=1)

        data['population_proper'] = data.apply(population_proper, axis=1)
        data.drop(columns=['df2_name'], inplace=True)

        geopopulation = data[[
            'lng', 'lat', 'capital', 'population', 'population_proper'
        ]]

        geopopulation.to_csv('data/geodata/geodata_1.csv', index=False)

    return data


def add_polar_coordinates(data, geo_population):
    console.log('Adding radius and angle')
    data['radius'] = np.sqrt((data['approximate_latitude']**2) +
                             (data['approximate_longitude']**2))
    data['angle'] = np.arctan2(data['approximate_longitude'],
                               data['approximate_latitude'])
    if geo_population:
        data['radius_city'] = np.sqrt((data['lat']**2) + (data['lng']**2))
        data['angle_city'] = np.arctan2(data['lng'], data['lat'])
    return data


def add_polar_pca(data, geo_population):
    """
    input: dataframe containing Latitude(x) and Longitude(y)
    """
    console.log('Adding polar PCA')
    coordinates = data[['approximate_latitude',
                        'approximate_longitude']].values
    pca_obj = PCA().fit(coordinates)
    pca_x = pca_obj.transform(
        data[['approximate_latitude', 'approximate_longitude']])[:, 0]
    pca_y = pca_obj.transform(
        data[['approximate_latitude', 'approximate_longitude']])[:, 1]

    data['geo_pca_x'] = pca_x
    data['geo_pca_y'] = pca_y

    # if geo_population:
    #     coordinates = data[['lat','lng']].values
    #     imp = SimpleImputer(strategy='mean')
    #     coordinates=imp.fit_transform(coordinates)
    #     pca_obj = PCA().fit(coordinates)
    #     pca_x = pca_obj.transform(coordinates)[:,0]
    #     pca_y = pca_obj.transform(coordinates)[:,1]

    #     data["geo_pca_x_city"]=pca_x
    #     data["geo_pca_y_city"]=pca_y
    return data


def orientation_x(name):
    name = str(name)
    if 'nan' in name:
        return 0
    if 'Est' in name:
        return 1
    if 'Ouest' in name:
        return -1


def orientation_y(name):
    name = str(name)
    if 'nan' in name:
        return 0
    if 'Nord' in name:
        return 1
    if 'Sud' in name:
        return -1


# Write a poem :


def caption(data, max_features=500):
    caption_df = pd.read_csv('data/image_captions/full_df.csv')

    caption_df.reset_index(drop=True, inplace=True)

    caption_df.drop(columns='id_annonce', inplace=True)

    caption_df['features'] = caption_df.apply(
        lambda row: row['features'].replace("['", '').replace("']", '').
        replace("'", ''),
        axis=1,
    )

    vec_tdidf = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer='word',  # stop_words=stop_words1,
        norm='l2',
        tokenizer=LemmaTokenizer(),
        max_features=max_features,
    )
    X_train_vect = vec_tdidf.fit_transform(caption_df['features'].values)
    X_train_sparse = pd.DataFrame.sparse.from_spmatrix(X_train_vect)
    X_train_sparse.columns = vec_tdidf.get_feature_names_out()


def preprocess(X_train_0, Y_train_0, X_test_0, parameters):
    """
    Data preprocessing pipeline
    """
    # Fixing seeds
    seed_everything()
    console.log('[bold green]Started preprocessing')

    # target transformation :
    if parameters['target_transformation']:
        Y_train_1 = np.log(Y_train_0)['price']
    else:
        Y_train_1 = Y_train_0

    # Kfold target encoding
    if not parameters['encoding']['target_encoding'] is None:
        X_train_enc, X_test_enc = kfold_target_encoder(
            X_train_0,
            X_test_0,
            Y_train_0,
            parameters['encoding']['target_encoding'],
            'price',
            folds=10,
        )
        X_train_0[parameters['encoding']['target_encoding']
                  [0]] = X_train_enc['_mean_enc']
        X_test_0[parameters['encoding']['target_encoding']
                 [0]] = X_test_enc['_mean_enc']

    # Concatenating data
    data = pd.concat([X_train_0, X_test_0], axis=0).reset_index(drop=True)

    # Adding geopopulation data
    if parameters['geo']['add_geopopulation']:
        data = add_geopopulation(data)

    if parameters['geo']['add_geopopulation_2']:
        data = add_geopopulation_2(data)

    # Dropping columns
    if not parameters['drop_columns'] is None:
        data = data.drop(columns=parameters['drop_columns'])

    # Add distance to city center
    if parameters['geo']['add_distance_to_city_center']:
        data = add_distance_to_center(data)

    # Frequency Encoding
    if not parameters['encoding']['frequency_encoding'] is None:
        for i in parameters['encoding']['frequency_encoding']:
            data[i] = data[i].apply(str)
            data = frequency_encoder(data, [i])

    # Label Encoding
    if not parameters['encoding']['label_encoding'] is None:
        for i in parameters['encoding']['label_encoding']:
            data[i] = data[i].apply(str)
            data = label_encoder(data, i)

    # Quantile Encoding
    if not parameters['encoding']['quantile_encoding'] is None:
        data = quantile_encoder(
            data,
            X_train_0,
            Y_train_0,
            X_test_0,
            parameters['encoding']['quantile_encoding'],
        )

    # Adding polar coordinates
    if parameters['polar']['add_polar_coordinates']:
        data = add_polar_coordinates(data,
                                     parameters['geo']['add_geopopulation'])

    # Adding polar rotation
    if not parameters['polar']['add_polar_rotation'] is None:
        data = add_polar_rotation(
            data,
            parameters['polar']['add_polar_rotation'],
            parameters['geo']['add_geopopulation'],
        )

    # adding geo pca data based on lat and long
    if parameters['polar']['add_polar_pca']:
        data = add_polar_pca(data, parameters['geo']['add_geopopulation'])

    if parameters['polar']['add_exposition_orientation']:
        data['orientation_x'] = data.apply(
            lambda row: orientation_x(row['exposition']), axis=1)
        data['orientation_y'] = data.apply(
            lambda row: orientation_x(row['exposition']), axis=1)

    # Constant imputation floor
    if parameters['imputation']['constant_imputation']['constant_floor']:
        data.loc[(data['property_type'] != 'appartement')
                 & (data['property_type'] != 'chambre')
                 & data['floor'].isna(), 'floor', ] = 0

    # Constant imputation land size
    if parameters['imputation']['constant_imputation']['constant_land_size']:
        data.loc[(data['property_type'] == 'chambre')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'duplex') &
                 (data['land_size'].isna()), 'land_size', ] = 0

    # constant imputation energy value
    if parameters['imputation']['constant_imputation'][
            'constant_energy_performance_value']:
        data.loc[(data['property_type'] == 'terrain')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'hôtel particulier')
                 & (data['energy_performance_value'].isna()),
                 'energy_performance_value', ] = 0

    # constant imputation ghg value
    if parameters['imputation']['constant_imputation']['constant_ghg_value']:
        data.loc[(data['property_type'] == 'terrain à bâtir')
                 | (data['property_type'] == 'péniche')
                 | (data['property_type'] == 'hôtel particulier')
                 & (data['ghg_value'].isna()), 'ghg_value', ] = 0

    # constant imputation bedrooms
    if parameters['imputation']['constant_imputation']['constant_bedrooms']:
        data.loc[(data['property_type'] == 'chambre') &
                 (data['nb_bedrooms'].isna()), 'nb_bedrooms', ] = 0

    # Constant imputation exposition
    if parameters['imputation']['constant_imputation']['constant_exposition']:
        data.loc[(data['property_type'] == 'péniche')
                 | (data['property_type'] == 'propriété')
                 | (data['property_type'] == 'parking')
                 | (data['property_type'] == 'terrain')
                 | (data['property_type'] == 'terrain à bâtir')
                 | (data['property_type'] == 'viager')
                 | (data['property_type'] == 'divers') &
                 (data['exposition'].isna()), 'exposition', ] = 0

    # Add geodata
    if parameters['geo']['add_geo']:
        data = add_geo(data, parameters['geo']['geodata'])

    # Adding images classification and quality
    if parameters['images']['nima']['add_classification_quality']:
        data = add_classification_quality(
            data,
            parameters['images']['nima']['images_features'],
            parameters['images']['nima']['classification_threshold'],
        )

    # Adding images extracted captions from transformer model
    if parameters['images']['caption']['add_captions']:
        console.log('Adding extracted captions from images')
        caption_df = pd.read_csv('data/image_captions/full_df.csv')
        caption_df.reset_index(drop=True, inplace=True)
        caption_df.drop(columns='id_annonce', inplace=True)
        for item in parameters['images']['caption']['has_items']:
            caption_df[f'Has_{item}'] = caption_df.apply(
                lambda row: int(row['features'].lower().count(item) > 0),
                axis=1)
        for item in parameters['images']['caption']['count_items']:
            caption_df[f'Count_{item}'] = caption_df.apply(
                lambda row: row['features'].lower().count(item), axis=1)

        caption_df.drop(inplace=True, columns=['features'], axis=1)
        data = pd.concat([data, caption_df], axis=1)

    # Hot encoding
    if parameters['encoding']['hot_encoding']:
        console.log('Hot Encoding')
        data = pd.get_dummies(data)

    # Mean Imputation
    if parameters['imputation']['mean_imputation']:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_0 = data.copy()
        imp_mean.fit(data)
        data = imp_mean.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # Minimum imputation
    if parameters['imputation']['mini_imputation']:
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
    if parameters['imputation']['iter_imputation']:
        iter_mean = IterativeImputer(estimator=LGBMRegressor(), random_state=0)
        data_0 = data.copy()
        iter_mean.fit(data)
        data = iter_mean.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    # Feautures interractions
    if parameters['features_interactions']:
        console.log('Adding polynomial features')
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

    if parameters['scaling']['standard_scaling']:
        console.log('Appling standard scaling')
        data_0 = data.copy()
        scaler.fit(data)
        data = scaler.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters['scaling']['robust_scaling']:
        console.log('Appling robust scaling')
        data_0 = data.copy()
        rbst_scaler.fit(data)
        data = rbst_scaler.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if parameters['scaling']['power_scaling']:
        console.log('Appling power scaling')
        data_0 = data.copy()
        power_transformer.fit(data)
        data = power_transformer.transform(data)
        data = pd.DataFrame(data, index=data_0.index, columns=data_0.columns)

    if 'id_annonce' in data.columns:
        data.drop(inplace=True, columns='id_annonce')

    if not parameters['final_dump'] is None:
        console.log(f"Dumping {parameters['final_dump']}")
        data.drop(inplace=True, columns=parameters['final_dump'])

    # Splitting the data back
    X_train_1 = data.loc[:X_train_0.index.max(), :]
    X_test_1 = data.loc[X_train_0.index.max() + 1:, :]

    console.log('[bold green]Finished preprocessing ')

    return X_train_1, Y_train_1, X_test_1
