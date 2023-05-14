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
    task = Task.create(project_name='real-estate-xgb', task_name=task_name)
    clearml_logger = task.get_logger()
    task.connect(preprocessing_parameters, 'preprocessing_parameters')

    N_FOLD = 10  # 10 for tests
    EARLY_STOP = 250
    XGB_ITERATIONS = 100

    list_predictors = []
    X_train_0, Y_train_0, X_test_0, X_test_ids, X_train_ids = load_data(
        'data/tabular/', False)
    xgb_params, lgb_params, cat_params = load_hyperparameters()
    X_train_1, Y_train_1, X_test_1 = preprocess(X_train_0, Y_train_0, X_test_0,
                                                preprocessing_parameters)
    console.log('Launched XGB Training : ')
    predictors = X_train_1.columns
    xgb_preds, xgb_train_preds, r2_xgb, mae_xgb, mse_xgb, rmse_xgb = train_xgb(
        X_train_1, Y_train_1, X_test_1)
    baseline_mae = mae_xgb
    console.log(f'columns {X_train_1.columns}')
    # i=0
    # # Current best for 100 iters and 10cv :
    # # R2 = 0.8062 --- MAE=0.2523 ---MSE=0.1268
    # while i<len(predictors) :
    #     i=0
    #     bad_columns =[]

    #     for col in predictors :
    #         console.log(f"Dropping : {col}")
    #         new_X_train =X_train_1.loc[:, ~X_train_1.columns.isin([col])]
    #         xgb_preds,xgb_train_preds, r2_xgb, mae_xgb, mse_xgb, rmse_xgb = train_xgb(new_X_train, Y_train_1,X_test_1)

    #         if mae_xgb<baseline_mae :
    #             bad_columns.append(col)
    #             X_train_1.drop(inplace=True, columns=col)
    #             console.log(f"New best score {mae_xgb} - baseline : {baseline_mae}")
    #             console.log(f"Removed column {col}")
    #             console.log(f"Total bad columns {bad_columns}")
    #             baseline_mae=mae_xgb
    #         else:
    #             i=i+1

    #         task.connect(xgb_params, name = "xgb_params")
    #         task.connect(lgb_params, name= "lgb_params")
    #         task.connect(cat_params, name = "cat_params")
    #         clearml_logger.report_single_value("XGB_ITERATIONS", XGB_ITERATIONS)
    #         clearml_logger.report_single_value("r2_xgb", r2_xgb)
    #         clearml_logger.report_single_value("mae_xgb", mae_xgb)
    #         clearml_logger.report_single_value("mse_xgb", mse_xgb)
    #         clearml_logger.report_single_value("rmse_xgb", rmse_xgb)
    #     predictors = list(set(predictors) - set(bad_columns))
