# Databricks notebook source
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, accuracy_score, log_loss, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import itertools

# Compatibility for databricks and jupyter notebook
def am_I_in_databricks():
    return 'databricks' in sys.executable

if not am_I_in_databricks():
    from IPython.display import display
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', -1)

def display_pandas(df):
    if am_I_in_databricks():
        display(spark.createDataFrame(df))
    else:
        display(df)
    
def display_pandas_str(df):
    if am_I_in_databricks():
        display_pandas(df.apply(lambda col: col.astype(str)))
    else:
        display(df)
        
def display_numpy(array, name):
    if am_I_in_databricks():
        display(spark.createDataFrame(pd.DataFrame(array, columns=[name])))
    else: 
        display(pd.DataFrame(array, columns=[name]))   
    
def display_plot():
    if am_I_in_databricks():
        display()
    else:
        plt.show()
    
# Excercise validation
def validate_assumptions(result, *assumptions):
    try:
        for assumption in assumptions:
            assert assumption[0](), assumption[1]
    except AssertionError as e:
        print('\n\x1b[0;31m\'Wrong: {}\x1b[0;0m\n'.format(e))
    else:
        print('\n\x1B[0;32mCorrect!\x1B[0m\n')
    finally:
        print('Your result is:\n\n')
        return result
    
# Useful transforms
def transform_column(df, column, transform):
    return pd.concat([df.drop(column, axis=1), pd.Series(transform(df[column]), name=column)], axis=1)
    
# Model evaluation
default_logistic_regression_grid = {'C': [200, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.0001, 0.00005, 0.00001]}
default_random_forest_grid = {'max_depth': [5, 8, 10, 15, 20, 30, 50]}

def normalized_confusion_matrix(x, y):
    conf_matrix = confusion_matrix(x, y)
    return (conf_matrix / conf_matrix.sum()).round(3)

metrics_map = {
    'accuracy': accuracy_score,
    'log_loss': log_loss,
    'f1_score': f1_score
}

def plot_normalized_confusion_matrix(normalized_matrix, classes, title):
    column_names = ['{}'.format(name) for name in classes]
    row_names = {i: '{}'.format(name) for i, name in enumerate(classes)}
        
    plt.clf()
    plt.figure(figsize=(len(classes) + 2, len(classes) + 2))
    plt.imshow(normalized_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(len(column_names)), column_names)
    plt.xlabel('Predicted')
    plt.yticks(np.arange(len(row_names)), row_names.values())
    plt.ylabel('True')
    
    for i, j in itertools.product(range(normalized_matrix.shape[0]), range(normalized_matrix.shape[1])):
        plt.text(j, i, format(normalized_matrix[i, j]),
                 horizontalalignment='center',
                 color='white' if normalized_matrix[i, j] > (normalized_matrix.max() / 2.) else 'black')

    display_plot()


def to_numpy(df):
    return df.as_matrix() if type(df) == pd.core.frame.DataFrame else df
    
    
def simple_classification_performance(raw_dataset, raw_labels, grid=None, 
    n_folds=10, oversampler=None, metrics=['accuracy'], model_name='logistic_regression', 
    n_estimators=100, run_grid_search=True):
    
    dataset = to_numpy(raw_dataset)
    labels = to_numpy(raw_labels)
    
    print('\nSelecting regularization parameter...')
    model_definition = (LogisticRegression(solver='sag', penalty='l2', tol=0.1, n_jobs=-1)
        if model_name == 'logistic_regression'
        else RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1))
    
    if model_name == 'logistic_regression' and not grid:
        grid = default_logistic_regression_grid
    elif not grid:
        grid = default_random_forest_grid
    
    optimal_hyperparameter = grid['C' if model_name == 'logistic_regression' else 'max_depth'][0]
    
    if run_grid_search:
        grid_search = GridSearchCV(
            model_definition, 
            grid,
#             scoring=make_scorer(f1_score, average='macro'),
            scoring='accuracy',
            n_jobs=-1)

        grid_search.fit(dataset, labels)

        optimal_hyperparameter = grid_search.best_params_['C' if model_name == 'logistic_regression' else 'max_depth']
    
    print('Using hyperparameter={}'.format(optimal_hyperparameter))
    hyperparametrized_model = (LogisticRegression(C=optimal_hyperparameter, penalty='l2', tol=0.1, solver='sag', n_jobs=-1)
        if model_name == 'logistic_regression'
        else RandomForestClassifier(max_depth=optimal_hyperparameter, n_estimators=n_estimators, n_jobs=-1))
        
    print('Evaluating model on {} folds...\n'.format(n_folds))   
    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=53)
    
    collected_metrics = {metric: [] for metric in metrics}
    auc_scores = []
    confusion_matrices = []
       
    for i, [train_index, test_index] in enumerate(k_fold.split(dataset, labels)):
        train_data, train_labels = ((dataset[train_index], labels[train_index])
            if not oversampler 
            else oversampler.fit_sample(dataset[train_index], labels.iloc[train_index]))

        model = hyperparametrized_model.fit(train_data, train_labels)
        prediction = model.predict(dataset[test_index])
        prediction_proba = model.predict_proba(dataset[test_index])[:, 1]
        
        test_labels = labels[test_index]
        
        confusion_matrices.append(confusion_matrix(test_labels, prediction))
        for metric in metrics:
            collected_metrics[metric].append(metrics_map[metric](test_labels, prediction))
    
    cf_matrix = np.array(confusion_matrices).mean(axis=0).astype(int)
    
    normalized_matrix = (cf_matrix / float(cf_matrix.sum())).round(3)
    
    plot_normalized_confusion_matrix(
        normalized_matrix, 
        model.classes_, 
        ''.join(['{}: {:.5f}'.format(metric, np.array(collected_metrics[metric]).mean())
            for metric in metrics]))
    
    return hyperparametrized_model

def check_misclassification(model, raw_dataset, raw_labels):
        dataset = to_numpy(raw_dataset)
        labels = to_numpy(raw_labels)
    
        train_features, test_features, train_labels, test_labels = train_test_split(
            dataset, labels, test_size=0.2, random_state=42)
        
#         predictions_train = model.predict(train_features)
        predictions_test = model.predict(test_features)
        
#         output_train = pd.concat(
#             [train_features, train_labels, pd.DataFrame(predictions_train, columns=['pred'], index=train_features.index)], 
#             axis=1)
        
#         output_test = pd.concat(
#             [test_features, test_labels, pd.DataFrame(predictions_test, columns=['pred'], index=test_features.index)], 
#             axis=1)
        
#         misclassified_train = output_train[predictions_train != train_labels]
        misclassified_test = test_features[predictions_test != test_labels]
        
#         return misclassified_train, misclassified_test
        return misclassified_test
