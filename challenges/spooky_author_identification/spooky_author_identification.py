import numpy as np
import pandas as pd
from hypotheses import RandomForrestClassifierHypothesis, LightGBMClassifierHypothesis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from category_encoders.hashing import HashingEncoder
from category_encoders import LeaveOneOutEncoder


def munge_features(df_train, df_test):
    df = pd.concat(objs=[df_train, df_test], axis=0, keys=['train', 'test'])

    # df['cabin_letter|categorical'] = df['cabin_id|string'].str.extract(pat=r'(\w)', expand=False).astype('category')
    # df['gender+maturity|categorical'] = df.apply(get_gender_maturity, axis=1).astype('category')
    # df['is_man|boolean'] = df['gender+maturity|categorical'] == 'man'
    # df['is_child|boolean'] = df['gender+maturity|categorical'].isin(['girl', 'boy']) | ((df['sex|binary'] == 'female') & (df['age|numerical'] <= 18))
    # df['group_id|string'] = (df['ticket_id|string'].str.slice(0, -1) + df['fare|numerical'].astype('str') + df['port_embarked|categorical'].astype('str'))
    # df['group_count|numerical'] = df.groupby('group_id|string')['group_id|string'].transform('count')


    required_cols = [

    ]
    return df.loc['train'][required_cols], df.loc['test'][required_cols]


def get_hypotheses(train, hyper_searcher_kwargs, cv_folds, cv_repeats):

    # SKLEARN RANDOM FOREST CLASSIFIER
    # ==================================================================================================================
    numerical = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ])
    string = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('target', LeaveOneOutEncoder(return_df=False, sigma=0)),
    ])
    categorical = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    ordinal = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ])
    binary = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ])
    boolean = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
    ])

    num_cols = [col for col in train.features if col.split('|')[-1] == 'numerical']
    string_cols = [col for col in train.features.columns if col.split('|')[-1] == 'string']
    cat_cols = [col for col in train.features.columns if col.split('|')[-1] == 'categorical']
    ordinal_cols = [col for col in train.features if col.split('|')[-1] == 'ordinal']
    binary_cols = [col for col in train.features if col.split('|')[-1] == 'binary']
    boolean_cols = [col for col in train.features if col.split('|')[-1] == 'boolean']

    transformer = ColumnTransformer(transformers=[
        ('num', numerical, num_cols),
        ('str', string, string_cols),
        ('cat', categorical, cat_cols),
        ('ord', ordinal, ordinal_cols),
        ('bin', binary, binary_cols),
        ('bool', boolean, boolean_cols),
    ])

    hyper_searcher_kwargs = {
        **hyper_searcher_kwargs,
        'cv': RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats),
        'scoring': 'accuracy',
        'error_score': 'raise',
        'refit': True
    }

    rf = RandomForrestClassifierHypothesis(
        dataset=train,
        transformer=transformer,
        hyper_searcher_strategy=RandomizedSearchCV,
        hyper_searcher_kwargs=hyper_searcher_kwargs,
        additional_hyper_dists={},
        n_trees=1024
    )


    # LIGHTGBM  CLASSIFIER
    # ==================================================================================================================
    lgb = LightGBMClassifierHypothesis(
        dataset=train,
        transformer=transformer,
        hyper_searcher_strategy=RandomizedSearchCV,
        hyper_searcher_kwargs=hyper_searcher_kwargs,
        additional_hyper_dists={},
        n_trees=256
    )

    # hypotheses = {
    #     'rf': rf,
    # }

    hypotheses = {
        'lgb': lgb,
    }

    return hypotheses
