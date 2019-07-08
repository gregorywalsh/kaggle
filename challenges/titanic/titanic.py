import numpy as np

from hypothesis import RandomForrestClassifierHypothesis
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold, KFold
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder


def initial_data_munge(df):
    return df[['sex|binary', 'port_embarked|categorical', 'class|ordinal']]


def get_hypotheses(train, test):

    def get_cat_ids(df):
        def cat_codes(s):  # def'd so can pickle
            return s.cat.codes
        return df.apply(cat_codes)

    # SKLEARN RANDOM FOREST CLASSIFIER
    cat_oh = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])
    cat_loo = Pipeline(steps=[
        ('leave_one_out', TargetEncoder(drop_invariant=True, smoothing=0.1))
    ])
    binary = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ])
    num = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ])
    cat_cols = [col for col in test.features.columns if col.split('|')[-1] == 'categorical']
    binary_cols = [col for col in test.features if col.split('|')[-1] == 'binary']
    ordinal_cols = [col for col in test.features if col.split('|')[-1] == 'ordinal']
    num_cols = [col for col in test.features if col.split('|')[-1] == 'numerical']

    oh_transformer = ColumnTransformer(transformers=[
        ('cat_oh', cat_oh, cat_cols),
        ('binary', binary, binary_cols),
        ('ord_encoder', binary, ordinal_cols),
        ('num', num, num_cols),
    ])

    loo_transformer = ColumnTransformer(transformers=[
        ('cat_loo', cat_loo, cat_cols),
        ('binary', binary, binary_cols),
        ('ord_encoder', binary, ordinal_cols),
        ('num', num, num_cols),
    ])

    rf_oh = RandomForrestClassifierHypothesis(
        dataset=train,
        pre_transformer=oh_transformer,
        n_trees=200
    )

    rf_loo = RandomForrestClassifierHypothesis(
        dataset=train,
        pre_transformer=loo_transformer,
        n_trees=200
    )

    rf_oh.hyperparam_dist = {'estimator__criterion': ['entropy'], 'estimator__max_depth': [32], 'estimator__max_features': [2], 'estimator__min_samples_split': [0.10641929674654432]}

    hypotheses = {
        'rf_oh': rf_oh,
        # 'rf_loo': rf_loo
    }
    return hypotheses


def get_scoring():
    return 'accuracy'


def get_cv():
    return RepeatedStratifiedKFold(n_splits=2, n_repeats=2)
