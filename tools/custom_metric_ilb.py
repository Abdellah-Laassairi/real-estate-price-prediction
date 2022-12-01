from sklearn.metrics import mean_absolute_percentage_error

def custom_metric_function(y_true, y_pred):
    return mean_absolute_percentage_error(y_true.price, y_pred.price)*100