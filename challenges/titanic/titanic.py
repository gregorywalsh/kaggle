import numpy as np

from hypothesis import RandomForrestClassifierHypothesis
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold


def initial_data_munge(df):
    return df[['sex|binary', 'port_embarked|categorical', 'class|ordinal']]


def get_hypotheses(train_features, test_features):

    def get_cat_ids(df):
        def cat_codes(s):  # def'd so can pickle
            return s.cat.codes
        return df.apply(cat_codes)

    # SKLEARN RANDOM FOREST CLASSIFIER
    cat_encoder = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])
    binary_encoder = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ])
    num_encoder = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ])
    pre_transformer = ColumnTransformer(transformers=[
        ('cat_encoder', cat_encoder, [col for col in test_features.columns if col.split('|')[-1] == 'categorical']),
        ('binary_encoder', binary_encoder, [col for col in test_features.columns if col.split('|')[-1] == 'binary']),
        ('ord_encoder', binary_encoder, [col for col in test_features.columns if col.split('|')[-1] == 'ordinal']),
        ('num_encoder', num_encoder, [col for col in test_features.columns if col.split('|')[-1] == 'numerical']),
    ])
    rf = RandomForrestClassifierHypothesis(
        features=train_features,
        pre_transformer=pre_transformer,
        n_trees=200
    )

    hypotheses = {
        'rf': rf
    }
    return hypotheses


def get_scoring():
    return 'accuracy'


def get_cv():
    return StratifiedKFold(5)
