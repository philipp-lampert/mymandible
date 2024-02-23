import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score, accuracy_score, average_precision_score, roc_auc_score, brier_score_loss

class threshold_optimized_metrics:
    
    def optimized_accuracy(y_test, y_pred):
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_acc = 0
        for threshold in thresholds:
            predicted_labels = (y_pred >= threshold).astype(int)
            acc = accuracy_score(y_test, predicted_labels)
            if acc > best_acc:
                best_acc = acc
        return best_acc

    def optimized_f1(y_test, y_pred):
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_f1 = 0
        for threshold in thresholds:
            predicted_labels = (y_pred >= threshold).astype(int)
            f1 = f1_score(y_test, predicted_labels)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1
    
    def optimized_mcc(y_test, y_pred):
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_mcc = -1
        for threshold in thresholds:
            predicted_labels = (y_pred >= threshold).astype(int)
            mcc = matthews_corrcoef(y_test, predicted_labels)
            if mcc > best_mcc:
                best_mcc = mcc
        return best_mcc
    

class preprocessing:
    
    def get_x_y(df, outcome, min_follow_up_days, scaler, drop_cols, inverse_pos):
        df = df.drop(columns=drop_cols)
        first_outcome_var = df.columns.get_loc('days_to_follow_up')
        predictors = df.columns[:first_outcome_var].tolist()

        data = df[df['days_to_follow_up'] >= min_follow_up_days].copy()
        data['days_to_flap_loss'] = data['days_to_flap_loss'].fillna(10000)
        data = data[data['days_to_flap_loss'] >= min_follow_up_days]
        data = data[predictors + [outcome]].dropna()

        if scaler != 'None':
            numeric_columns = data[predictors].select_dtypes(np.number).columns.tolist()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Flip outcome values if n(positive class) > n(negative class)
        if inverse_pos == True:
            value_counts = data[outcome].value_counts()
            if value_counts[True] > value_counts[False]:
                data[outcome] = ~data[outcome]
                warnings.warn("\nThe outcome variable has been inversed due to positive values being the majority class. This may lead to misinterpretation of coefficients and/or feature importances but keeps metrics such as F1-scores comparable.\n")

        return data[predictors], data[outcome]
    
