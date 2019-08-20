import collections
import yaml
import pandas as pd


PANDAS_DTYPE_MAP = {
    'unique': 'object',
    'string': 'object',
    'date': 'datetime64',
    'binary': 'category',
    'bool': 'boolean',
    'categorical': 'category',
    'categorical_low': 'category',
    'categorical_hi': 'category',
    'ordinal': pd.api.types.CategoricalDtype,
    'numerical': 'float64',
}


class ColumnDefn:

    def __init__(self, name, vartype, group, load, friendly_name=None, ordinal_elements=None):
        self.name = name
        self.new_name = '{}|{}'.format(friendly_name if friendly_name else name, vartype)
        self.vartype = vartype
        self.group = group
        self.load = load
        self.ordinal_elements = ordinal_elements
        if vartype == 'ordinal':
            self.dtype = PANDAS_DTYPE_MAP[self.vartype](categories=ordinal_elements, ordered=True)
        else:
            self.dtype = PANDAS_DTYPE_MAP[self.vartype]


def load_yaml_config(fp):
    with open(fp, 'r') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return raw_config


def csv_to_df(config_fp, data_fp, num_rows=None, is_test=True):

    # Load column definitions
    config = load_yaml_config(fp=config_fp)
    col_defns = [ColumnDefn(**kwargs) for kwargs in config['column_defns']
                 if not (is_test and kwargs['group'] == 'target')]

    # Validate the required and source column names
    req_col_names = list(defn.name for defn in col_defns)
    source_col_names = pd.read_csv(
        filepath_or_buffer=data_fp,
        nrows=0,
        **config['pandas_csv_kwargs']
    ).columns
    if len(set(source_col_names)) != len(source_col_names):
        raise ValueError(
            'Source data contains columns with duplicate names.({duplicates})'.format(
                duplicates=[item for item, count in collections.Counter(source_col_names).items() if count > 1]
            )
        )
    if len(set(req_col_names)) != len(req_col_names):
        raise ValueError(
            'Arg "required_cols" contains columns with duplicate names.({duplicates})'.format(
                duplicates=[item for item, count in collections.Counter(req_col_names).items() if count > 1]
            )
        )
    if not set(req_col_names).issubset(set(source_col_names)):
        raise ValueError(
            '"{missed_cols}" in "required_col_names" were not found in the source columns'.format(
                missed_cols=set(req_col_names).difference(set(source_col_names))
            )
        )

    # Read the CSV
    date_col_names = [defn.name for defn in col_defns if defn.dtype == 'datetime64' and defn.load]
    dtype_map = {defn.name: defn.dtype for defn in col_defns if not defn.dtype == 'datetime64' and defn.load}
    df = pd.read_csv(
        filepath_or_buffer=data_fp,
        usecols=req_col_names,
        dtype=dtype_map,
        parse_dates=date_col_names,
        nrows=num_rows,
        **config['pandas_csv_kwargs']
    )

    # Update df column names
    col_defs_by_name = {defn.name: defn for defn in col_defns if defn.load}
    new_col_names = [col_defs_by_name[name].new_name for name in df.columns]
    df.columns = new_col_names

    # Create convenience views into the data
    meta_col_names = [defn.new_name for defn in col_defns if defn.group == 'meta']
    feature_col_names = [defn.new_name for defn in col_defns if defn.group == 'feature']
    target_col_names = [defn.new_name for defn in col_defns if defn.group == 'target']
    meta = df[meta_col_names] if meta_col_names else None
    features = df[feature_col_names] if feature_col_names else None
    target = df[target_col_names] if target_col_names else None

    return meta, features, target, df


def print_title(title):
    many_stars = '*' * 100
    print(many_stars + '\n' + title + '\n' + many_stars)


def process_cv_results(run_id, hypothesis_name, cv_results, cv_folds, cv_repeats, num_hyp_samples):
    df = pd.DataFrame.from_dict(data=cv_results)
    df['run_id'] = run_id
    df['hypothesis_name'] = hypothesis_name
    df['cv_folds'] = cv_folds
    df['cv_repeats'] = cv_repeats
    df['num_hyp_samples'] = num_hyp_samples
    df['min_score'] = df.filter(regex=r'^split\d+_test_score').min(axis=1)
    df['max_score'] = df.filter(regex=r'^split\d+_test_score').max(axis=1)
    df.sort_values(by='rank_test_score', ascending=True, inplace=True)
    return df


def save_cv_results(path, processed_cv_results, reporting_keys, top_n=None):
    with open(file=path, mode='a') as f:
        processed_cv_results[reporting_keys].iloc[0:top_n].to_csv(path_or_buf=f, index=False, header=False)
    return


def save_model_repr(path, model):
    with open(file=path, mode='a+') as f:
        f.write(model.__repr__(float('inf')))  # Get all chars with float('inf')


def print_df_details(df, name):

    print('=' * 100, '\n')
    print('Details of ' + name, '\n')
    print("Number of rows in train dataset : ", df.shape[0], '\n')
    print("Head:")
    print(df.head(5), '\n')
    print("Tail:")
    print(df.tail(5), '\n')
    print(df.describe(), '\n')
    print(df.info(), '\n')
    print('='*100)
