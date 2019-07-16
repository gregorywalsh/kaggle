import numpy as np
from hypotheses import RandomForrestClassifierHypothesis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from category_encoders.hashing import HashingEncoder


def munge_features(df):
    df['title|categorical'] = df['name|unique'].str.extract(pat='(?<=, )(\\w+)', expand=False).astype('category')
    df['surname|string'] = df['name|unique'].str.extract(pat='([\\w'']+)')
    df['in_family|boolean'] = (df['num_sib_spouses|numerical'] > 0) & (df['num_parents_children|numerical'] > 0).astype('float64')
    df['has_cabin|boolean'] = (~df['cabin_id|string'].isnull()).astype('float64')
    df['cabin_letter|categorical'] = df['cabin_id|string'].str.extract(pat='(\\w)', expand=False).astype('category')
    df['cabin_number|numerical'] = df['cabin_id|string'].str.extract(pat='(\\d+)').astype(np.float64)
    return df.drop(columns=['name|unique'])


def get_cat_ids(df):
    def cat_codes(s):  # def'd rather than lambda so can pickle, FYI cat code -1 is for np.NaN
        return s.cat.codes
    return df.apply(cat_codes)


def get_hypotheses(train, hyper_searcher_kwargs):

    # SKLEARN RANDOM FOREST CLASSIFIER
    numerical = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ])
    string = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('hashing', HashingEncoder(return_df=False, n_components=10))
    ])
    categorical = Pipeline(steps=[
        ('cat_codes', FunctionTransformer(func=get_cat_ids, validate=False)),
        ('imputer', SimpleImputer(missing_values=-1, strategy='most_frequent')),
        # ('target', TargetEncoder(return_df=False, smoothing=0.5)),
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
        'cv': RepeatedStratifiedKFold(n_splits=5, n_repeats=1),
        'scoring': 'accuracy',
        'error_score': 'raise',
        'refit': True
    }

    additional_hyper_dists = {
        # 'transformer__cat__target__smoothing': [0.25, 0.5, 0.75],
        'transformer__str__hashing__n_components': [1, 2, 4, 8, 16, 32]
    }

    rf = RandomForrestClassifierHypothesis(
        dataset=train,
        transformer=transformer,
        hyper_searcher_strategy=RandomizedSearchCV,
        hyper_searcher_kwargs=hyper_searcher_kwargs,
        additional_hyper_dists=additional_hyper_dists,
        n_trees=200
    )

    hypotheses = {
        'rf': rf,
    }

    return hypotheses
