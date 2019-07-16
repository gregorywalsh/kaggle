import pandas as pd
import csv
import collections
import yaml

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
    yaml_tag = u'!ColumnDefn'

    def __init__(self, name, vartype, group, load, friendly_name=None, ordinal_elements=None):
        self.name = name
        self.new_name = '{}|{}'.format(friendly_name if friendly_name else name, vartype)
        self.vartype = vartype
        self.ordinal_elements = ordinal_elements
        self.group = group
        if vartype == 'ordinal':
            self.dtype = PANDAS_DTYPE_MAP[self.vartype](categories=ordinal_elements, ordered=True)
        else:
            self.dtype = PANDAS_DTYPE_MAP[self.vartype]
        self.load = load

    def __repr__(self):
        return '{c}(name={n}, new_name:{m}, vartype:{v}, group={g}, dtype={d}, loaded={l}, ordinals={o}}'.format(
            c=self.__class__.__name__, n=self.name, m=self.new_name, v=self.vartype,
            g=self.group, d=self.dtype, l=self.load, o=self.ordinal_elements
        )


class Dataset:

    def __init__(self, config_fp, data, num_rows, is_test=False, always_validate=True, verbose=False):

        self._is_first_load = True
        self._is_test = is_test
        self._verbose = verbose
        self._always_validate = always_validate

        pandas_csv_kwargs, self.col_defns = self.load_config(config_fp)
        pandas_csv_kwargs['nrows'] = num_rows
        self.meta, self.features, self.target = self.load_from_csv(
            filepath=data,
            col_defns=self.col_defns,
            pandas_csv_kwargs=pandas_csv_kwargs
        )

        # Update flags
        self._is_first_load = False

    def load_config(self, config_fp):

        with open(config_fp, 'r') as f:
            try:
                raw_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        col_defns = [ColumnDefn(**kwargs) for kwargs in raw_config['column_defns']
                     if not (self._is_test and kwargs['group'] == 'target')  # Prevent train from loading target cols
                     ]
        return raw_config['pandas_csv_kwargs'], col_defns

    def load_from_csv(self, filepath, col_defns, pandas_csv_kwargs):

        # Validate the column names
        col_names = list(defn.name for defn in col_defns)
        self._check_cols(source_col_names=Dataset._get_file_columns(filepath), col_names=col_names)

        # Create a Dataframe from the CSV
        date_col_names = [defn.name for defn in col_defns if defn.dtype == 'datetime64' and defn.load]
        dtype_map = {defn.name: defn.dtype for defn in col_defns if not defn.dtype == 'datetime64' and defn.load}
        df = pd.read_csv(
            filepath_or_buffer=filepath,
            usecols=col_names,
            dtype=dtype_map,
            parse_dates=date_col_names,
            **pandas_csv_kwargs
        )

        col_defs_by_name = {defn.name: defn for defn in col_defns if defn.load}
        new_col_names = [col_defs_by_name[name].new_name for name in df.columns]
        df.columns = new_col_names

        # Return
        return self._create_df_views(df=df, col_defns=col_defns)

    # def load_from_list(self, list_of_tuples, source_col_names):
    # TODO: Reintegrate loading from a list of tuples
    #
    #     # Validate the column names
    #     self._check_cols(source_col_names=source_col_names)
    #
    #     # Create a Dataframe from the list of tuples
    #     df = pd.DataFrame(
    #         data=list_of_tuples,
    #         columns=source_col_names,
    #     )
    #     self._create_df_views(df)

    def _check_cols(self, source_col_names, col_names):

        if self._always_validate or self._is_first_load:
            if len(set(source_col_names)) != len(source_col_names):
                raise ValueError(
                    'Source data contains columns with duplicate names.({duplicates})'.format(
                        duplicates=[item for item, count in collections.Counter(source_col_names).items() if count > 1]
                    )
                )

            if len(set(col_names)) != len(col_names):
                raise ValueError(
                    'Arg "required_cols" contains columns with duplicate names.({duplicates})'.format(
                        duplicates=[item for item, count in collections.Counter(col_names).items() if count > 1]
                    )
                )

            if not set(col_names).issubset(set(source_col_names)):
                raise ValueError(
                    '"{missed_cols}" in "required_col_names" were not found in the source columns'.format(
                        missed_cols=set(col_names).difference(set(source_col_names))
                    )
                )

    def _create_df_views(self, df, col_defns):

        # Determine the variable types of the cols
        if self._always_validate or self._is_first_load:
            self.meta_col_names = [defn.new_name for defn in col_defns if defn.group == 'meta']
            self.feature_col_names = [defn.new_name for defn in col_defns if defn.group == 'feature']
            self.target_col_names = [defn.new_name for defn in col_defns if defn.group == 'target']

        # Create convenience views into the data
        meta = df[self.meta_col_names] if self.meta_col_names else None
        features = df[self.feature_col_names] if self.feature_col_names else None
        target = df[self.target_col_names] if self.target_col_names else None

        return meta, features, target


    @staticmethod
    def _get_file_columns(filepath):
        # TODO: Use pandas csv loader here instead so kwargs can be passed for the sake of consistency
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            columns = next(reader)
        return columns
