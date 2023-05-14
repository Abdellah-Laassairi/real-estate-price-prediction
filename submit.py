import random
import string
import time

import catboost as cat
import xgboost as xgb
import yaml
from catboost import CatBoostRegressor
from clearml import Task
from lightgbm import LGBMRegressor
from rich.console import Console
from rich.progress import *
from sklearn.ensemble import *
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import *
from xgboost import plot_importance
from xgboost import XGBRegressor
from yaml.loader import SafeLoader

from tools.preprocess import *
from tools.selector import *


def train_xgb(X_train, y_train, X_test):
    nfold = N_FOLD
    skf = KFold(n_splits=nfold, shuffle=True, random_state=2019)
    progress_bar = Progress(
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TimeRemainingColumn(),
    )

    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_train))
    final_predictions = np.zeros(len(X_test))

    predictors = X_train.columns.values.tolist()

    i = 1

    with progress_bar as progress:
        task1 = progress.add_task('[red]Training XGB', total=10)
        while not progress.finished:

            for train_index, valid_index in skf.split(X_train, y_train.values):
                # print("\nFold {}".format(i))
                xg_train = xgb.DMatrix(
                    X_train.iloc[train_index][predictors].values,
                    y_train.iloc[train_index].values,
                )
                xg_valid = xgb.DMatrix(
                    X_train.iloc[valid_index][predictors].values,
                    y_train.iloc[valid_index].values,
                )

                clf = xgb.train(xgb_params,
                                xg_train,
                                XGB_ITERATIONS,
                                evals=[(xg_train, 'train'),
                                       (xg_valid, 'eval')],
                                verbose_eval=False)
                oof[valid_index] = clf.predict(
                    xgb.DMatrix(X_train.iloc[valid_index][predictors].values))

                predictions += clf.predict(
                    xgb.DMatrix(X_train[predictors].values)) / nfold
                final_predictions += clf.predict(
                    (xgb.DMatrix(X_test[predictors].values))) / nfold

                i = i + 1
                progress.update(task1, advance=1)
                time.sleep(0.5)
    R2 = r2_score(y_train.values, oof)
    mae = mean_absolute_error(y_train.values, oof)
    mse = mean_squared_error(y_train.values, oof)
    rmse = np.sqrt(mse)
    console.log(
        f'R2 = {R2:<0.4f} --- MAE={mae:<0.4f} ---MSE={mse:<0.4f} --- RMSE ={rmse:<0.4f} | CV ={N_FOLD}'
    )
    return final_predictions, predictions, R2, mae, mse, rmse


def train_lgb(X_train, y_train, X_test):
    nfold = N_FOLD
    skf = KFold(n_splits=nfold, shuffle=True, random_state=2019)
    progress_bar = Progress(
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TimeRemainingColumn(),
    )
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_train))
    final_predictions = np.zeros(len(X_test))

    predictors = X_train.columns.values.tolist()

    i = 1
    with progress_bar as progress:
        task1 = progress.add_task('[red]Training LGB', total=10)
        while not progress.finished:
            for train_index, valid_index in skf.split(X_train, y_train.values):
                d_train = lgb.Dataset(
                    X_train.iloc[train_index][predictors].values,
                    y_train.iloc[train_index].values,
                )
                d_valid = lgb.Dataset(
                    X_train.iloc[valid_index][predictors].values,
                    y_train.iloc[valid_index].values,
                )
                watchlist = [d_valid]
                clf = lgb.train(lgb_params,
                                d_train,
                                num_boost_round=LGB_ITERATIONS,
                                valid_sets=d_valid,
                                callbacks=[lgb.log_evaluation(period=0)])
                oof[valid_index] = clf.predict(
                    X_train.iloc[valid_index][predictors].values)

                predictions += clf.predict(X_train[predictors].values) / nfold
                final_predictions += clf.predict(
                    X_test[predictors].values) / nfold
                i = i + 1
                progress.update(task1, advance=1)
                time.sleep(0.5)

    R2 = r2_score(y_train.values, oof)
    mae = mean_absolute_error(y_train.values, oof)
    mse = mean_squared_error(y_train.values, oof)
    rmse = np.sqrt(mse)
    console.log(
        f'R2 = {R2:<0.4f} --- MAE={mae:<0.4f} ---MSE={mse:<0.4f} --- RMSE ={rmse:<0.4f} | CV ={N_FOLD}'
    )
    return final_predictions, predictions, R2, mae, mse, rmse


def train_cat(X_train, y_train, X_test):
    nfold = N_FOLD
    skf = KFold(n_splits=nfold, shuffle=True, random_state=2019)
    progress_bar = Progress(
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('•'),
        TimeElapsedColumn(),
        TextColumn('•'),
        TimeRemainingColumn(),
    )
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_train))
    final_predictions = np.zeros(len(X_test))
    predictors = X_train.columns.values.tolist()

    i = 1
    with progress_bar as progress:
        task1 = progress.add_task('[red]Training CAT', total=10)
        while not progress.finished:
            for train_index, valid_index in skf.split(X_train, y_train.values):
                # print("\nFold {}".format(i))
                d_train = cat.Pool(
                    X_train.iloc[train_index][predictors].values,
                    y_train.iloc[train_index].values,
                )
                d_valid = cat.Pool(
                    X_train.iloc[valid_index][predictors].values,
                    y_train.iloc[valid_index].values,
                )
                watchlist = [d_valid]
                clf = cat.train(d_train,
                                cat_params,
                                num_boost_round=CAT_ITERATIONS,
                                eval_set=d_valid,
                                early_stopping_rounds=EARLY_STOP,
                                verbose_eval=0)
                oof[valid_index] = clf.predict(
                    X_train.iloc[valid_index][predictors].values)

                predictions += clf.predict(X_train[predictors].values) / nfold
                final_predictions += clf.predict(
                    X_test[predictors].values) / nfold

                i = i + 1
                progress.update(task1, advance=1)
                time.sleep(0.5)
    R2 = r2_score(y_train.values, oof)
    mae = mean_absolute_error(y_train.values, oof)
    mse = mean_squared_error(y_train.values, oof)
    rmse = np.sqrt(mse)
    console.log(
        f'R2 = {R2:<0.4f} --- MAE={mae:<0.4f} ---MSE={mse:<0.4f} --- RMSE ={rmse:<0.4f} | CV ={N_FOLD}'
    )
    return final_predictions, predictions, R2, mae, mse, rmse


letters = string.digits
buzzwords = ['khrba', 'dar', 'hmza', 'mdrasa', 'ard']
morewords = ['kbira', 'sghira', 'zwina', 'khayba', 'jdida', '9dima']
task_name = f'{random.choice(letters)}-{random.choice(buzzwords)}-{random.choice(morewords)}'

if __name__ == '__main__':
    with open('preprocess.yaml', 'r') as f:
        preprocessing_parameters = yaml.load(f, Loader=SafeLoader)
    console = Console()
    # Creating task & connecting preprocessing parameters
    console.log(f'[bold green]Creating Task {task_name}')
    task = Task.create(project_name='real-estate', task_name=task_name)
    clearml_logger = task.get_logger()
    task.connect(preprocessing_parameters, 'preprocessing_parameters')

    X_train_0, Y_train_0, X_test_0, X_test_ids, X_train_ids = load_data(
        'data/tabular/', False)

    xgb_params, lgb_params, cat_params = load_hyperparameters()
    X_train_1, Y_train_1, X_test_1 = preprocess(X_train_0, Y_train_0, X_test_0,
                                                preprocessing_parameters)

    N_FOLD = 50  # 10 for tests
    EARLY_STOP = 250
    # XGB_ITERATIONS = 100
    # LGB_ITERATIONS = 250
    # CAT_ITERATIONS = 300
    XGB_ITERATIONS = 1000
    LGB_ITERATIONS = 2000
    CAT_ITERATIONS = 3000
    task.connect(xgb_params, name='xgb_params')
    task.connect(lgb_params, name='lgb_params')
    task.connect(cat_params, name='cat_params')
    clearml_logger.report_single_value('XGB_ITERATIONS', XGB_ITERATIONS)
    clearml_logger.report_single_value('LGB_ITERATIONS', LGB_ITERATIONS)
    clearml_logger.report_single_value('CAT_ITERATIONS', CAT_ITERATIONS)

    console.log('Launched XGB Training : ')
    xgb_preds, xgb_train_preds, r2_xgb, mae_xgb, mse_xgb, rmse_xgb = train_xgb(
        X_train_1, Y_train_1, X_test_1)
    clearml_logger.report_single_value('r2_xgb', r2_xgb)
    clearml_logger.report_single_value('mae_xgb', mae_xgb)
    clearml_logger.report_single_value('mse_xgb', mse_xgb)
    clearml_logger.report_single_value('rmse_xgb', rmse_xgb)
    # R2 = 0.8123 --- MAE=0.2419 ---MSE=0.1228 --- RMSE =0.3504 | CV =50 | 1000 iters

    console.log('Launched LGB Training')
    lgb_preds, lgb_train_preds, r2_lgb, mae_lgb, mse_lgb, rmse_lgb = train_lgb(
        X_train_1, Y_train_1, X_test_1)
    clearml_logger.report_single_value('r2_lgb', r2_lgb)
    clearml_logger.report_single_value('mae_lgb', mae_lgb)
    clearml_logger.report_single_value('mse_lgb', mse_lgb)
    clearml_logger.report_single_value('rmse_lgb', rmse_lgb)

    # R2 = 0.8216 --- MAE=0.2328 ---MSE=0.1167 --- RMSE =0.3416 CV10
    # R2 = 0.8252 --- MAE=0.2310 ---MSE=0.1144 --- RMSE =0.3382 CV25
    # R2 = 0.8272 --- MAE=0.2274 ---MSE=0.1130 --- RMSE =0.3362 | CV =50 | 2000 iters

    console.log('Launched cat training')
    cat_preds, cat_train_preds, r2_cat, mae_cat, mse_cat, rmse_cat = train_cat(
        X_train_1, Y_train_1, X_test_1)
    clearml_logger.report_single_value('r2_cat', r2_cat)
    clearml_logger.report_single_value('mae_cat', mae_cat)
    clearml_logger.report_single_value('mse_cat', mse_cat)
    clearml_logger.report_single_value('rmse_cat', rmse_cat)

    # R2 = 0.8188 --- MAE=0.2386 ---MSE=0.1185 --- RMSE =0.3443 CV 10
    # R2 = 0.8252 --- MAE=0.2340 ---MSE=0.1143 --- RMSE =0.3381 | CV =50 | iters 3000

    console.log('Submitting...')
    weights = [5 / 20, 9 / 20, 6 / 20]

    final_predictions_sum = (weights[0] * np.exp(xgb_preds) +
                             weights[1] * np.exp(lgb_preds) +
                             weights[2] * np.exp(cat_preds))

    final_predictions = pd.Series(final_predictions_sum, name='price')
    final_predictions.head()
    final_submission = pd.concat([X_test_ids, final_predictions], axis=1)
    final_submission['id_annonce'] = final_submission['id_annonce'].astype(
        np.int32)
    final_submission.to_csv('data/final_submission_168.csv',
                            index=False,
                            header=True)
    console.log('Finished submitting')

    train_sum = (weights[0] * np.exp(xgb_train_preds) +
                 weights[1] * np.exp(lgb_train_preds) +
                 weights[2] * np.exp(cat_train_preds))

    train_predictions = pd.Series(train_sum, name='price')
    train_predictions.head()
    train_submission = pd.concat([X_train_ids, train_predictions], axis=1)
    train_submission['id_annonce'] = train_submission['id_annonce'].astype(
        np.int32)
    train_submission.to_csv('data/train_submission.csv',
                            index=False,
                            header=True)
    console.log('Finished submitting for training data')
