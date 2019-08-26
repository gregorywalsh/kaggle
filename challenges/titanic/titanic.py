import numpy as np
import pandas as pd
from hypothesis import RandomForrestClassifierHypothesis, LightGBMClassifierHypothesis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from category_encoders.hashing import HashingEncoder
from category_encoders import LeaveOneOutEncoder

# def munge_features(df):
#     df['title|categorical'] = df['name|unique'].str.extract(pat=r'(?<=, )(\w+)', expand=False).astype('category')
#     df['surname|string'] = df['name|unique'].str.extract(pat=r'([\w'']+)')
#     df['in_family|boolean'] = ((df['num_sib_spouses|numerical'] > 0) & (df['num_parents_children|numerical'] > 0)).astype('float64')
#     df['has_cabin|boolean'] = (~df['cabin_id|string'].isnull()).astype('float64')
#     df['cabin_letter|categorical'] = df['cabin_id|string'].str.extract(pat=r'(\w)', expand=False).astype('category')
#     df['cabin_number|numerical'] = df['cabin_id|string'].str.extract(pat=r'(\d+)').astype(np.float64)
#     return df.drop(columns=['name|unique'])


# def get_gender_maturity(row):
#     if row['sex|binary'] == 'female':
#         if row['age|numerical'] <= 12:
#             gm = 'girl'
#         else:
#             gm = 'woman'
#
#     elif row['title|categorical'] == 'Master':
#         gm = 'boy'
#     else:
#         gm = 'man'
#     return gm


def get_gender_maturity(row):
    if row['sex|binary'] == 'female':
        gm = 'female'
    elif row['title|categorical'] == 'Master':
        gm = 'boy'
    else:
        gm = 'man'
    return gm


def munge_features(df_train, df_test):
    df = pd.concat(objs=[df_train, df_test], axis=0, keys=['train', 'test'])

    df['cabin_letter|categorical'] = df['cabin_id|string'].str.extract(pat=r'(\w)', expand=False).astype('category')
    df['cabin_number|numerical'] = df['cabin_id|string'].str.extract(pat=r'(\d+)').astype(np.float64)

    df['title|categorical'] = df['name|unique'].str.extract(pat=r'(?<=, )(\w+)', expand=False).astype('category')
    df['gender+maturity|categorical'] = df.apply(get_gender_maturity, axis=1).astype('category')
    df['surname|string'] = df['name|unique'].str.extract(pat=r'([\w'']+)')

    df['is_man|boolean'] = df['gender+maturity|categorical'] == 'man'
    df['is_female|boolean'] = df['gender+maturity|categorical'] == 'female'
    df['is_child|boolean'] = df['gender+maturity|categorical'].isin(['girl', 'boy']) | ((df['sex|binary'] == 'female') & (df['age|numerical'] <= 18))

    df['group_id|string'] = (df['ticket_id|string'].str.slice(0, -1) + df['fare|numerical'].astype('str') + df['port_embarked|categorical'].astype('str'))
    df['group_count|numerical'] = df.groupby('group_id|string')['group_id|string'].transform('count')
    df['group_has_man|boolean'] = df.groupby('group_id|string')['is_man|boolean'].transform('any').astype('float64')
    df['group_has_child|boolean'] = df.groupby('group_id|string')['is_child|boolean'].transform('any').astype('float64')
    df['group_count_men|numerical'] = df.groupby('group_id|string')['is_man|boolean'].transform('sum').astype('float64')
    df['group_count_child|numerical'] = df.groupby('group_id|string')['is_child|boolean'].transform('sum').astype('float64')
    df['group_count_female|numerical'] = df.groupby('group_id|string')['is_female|boolean'].transform('sum').astype('float64')
    df['group_prop_child|numerical'] = df['group_count_child|numerical'] / df['group_count|numerical']
    df['group_avg_fare|numerical'] = df['fare|numerical'] / df['group_count|numerical']

    df['is_woman_or_child|boolean'] = df['is_child|boolean'] | df['is_female|boolean']
    df['wc_group_id|string'] = df['group_id|string'] + df['is_woman_or_child|boolean'].astype('str')
    df['wc_group_count|numerical'] = df.groupby('wc_group_id|string')['wc_group_id|string'].transform('count')

    required_cols = [
        # NOT SO GOOD FEATURES
        # 'sex|binary',
        # 'port_embarked|categorical',
        # 'group_has_man|numerical',
        # 'group_prop_women_child|numerical',
        # 'group_prop_child|numerical',
        # 'group_avg_fare|numerical',
        # 'group_count_female|numerical',

        # # GOOD GROUP FEATURES
        # 'class|ordinal',
        # 'gender+maturity|categorical',
        # 'group_id|string',
        # 'group_count|numerical',
        # # 'group_count_men|numerical',
        # # 'group_count_child|numerical', (works well on train but not on test...)

        # GOOD WC GROUP FEATURES
        'class|ordinal',
        'gender+maturity|categorical',
        'wc_group_id|string',
        'wc_group_count|numerical',
        # 'group_count_men|numerical',
    ]
    return df.loc['train'][required_cols], df.loc['test'][required_cols]


def get_cat_ids(df):
    def cat_codes(s):  # def'd rather than lambda so can pickle, FYI cat code -1 is for np.NaN
        return s.cat.codes
    return df.apply(cat_codes)


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
        x=train,
        cv_transformer=transformer,
        hyper_searcher_strategy=RandomizedSearchCV,
        hyper_searcher_kwargs=hyper_searcher_kwargs,
        additional_hyper_dists={},
        n_trees=1024
    )


    # LIGHTGBM  CLASSIFIER
    # ==================================================================================================================
    lgb = LightGBMClassifierHypothesis(
        x=train,
        transformer=transformer,
        hyper_searcher_strat=RandomizedSearchCV,
        hyper_searcher_kwargs=hyper_searcher_kwargs,
        additional_hyper_dists={},
        n_trees=256
    )

    # saved_hypotheses = {
    #     'rf': rf,
    # }

    hypotheses = {
        'lgb': lgb,
    }

    return hypotheses
