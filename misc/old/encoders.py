import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def get_simple_imputer(test_feature, train_feature):
    if np.isnan(x=train_feature).any():
        steps = [('impute', SimpleImputer(strategy='constant'))]
    else:
        if np.isnan(x=test_feature).any():
            steps = [('impute', SimpleImputer(strategy='most_frequent'))]
        else:
            steps = []
    return steps


def _choose_imputer(imputer_name, test_feature, train_feature):
    if imputer_name == 'simple':
        imputer = get_simple_imputer(test_feature=test_feature, train_feature=train_feature)
    else:
        raise ValueError('"{c}" is not a valid choice of imputer'.format(c=imputer_name))
    # TODO - Add more imputing options
    return imputer


def get_categorical_embedding_pipeline(test_feature, train_feature, imputer=None):
    if imputer:
        steps = _choose_imputer(imputer_name=imputer, test_feature=test_feature, train_feature=train_feature)
    else:
        steps = []

    steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    return Pipeline(steps=steps)


def get_categorical_embedding_pipeline(test_feature, train_feature, missing_values=('',)):
    # Check if there are any missing values in the test or trin feature and recommend an appropriate embedding
    missing_values = missing_values + (np.NAN,)  # NOTE - np.isin does not work for NaN hence np.isnan check
    if np.isin(element=train_feature, test_elements=missing_values).any() or np.isnan(x=train_feature).any():
        steps = [('impute', SimpleImputer(missing_values=missing_values, strategy='constant'))]
    else:
        if np.isin(element=test_feature, test_elements=missing_values).any() or np.isnan(x=test_feature).any():
            steps = [('impute', SimpleImputer(missing_values=missing_values, strategy='most_frequent'))]
        else:
            steps = []
    steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    return Pipeline(steps=steps)