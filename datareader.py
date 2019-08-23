import pandas as pd
import csv
import collections
from util import load_yaml_config

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

    def __init__(self, name, vartype, group, required, friendly_name=None, ordinal_elements=None):
        self.name = name
        self.new_name = '{}|{}'.format(friendly_name if friendly_name else name, vartype)
        self.vartype = vartype
        self.ordinal_elements = ordinal_elements
        self.group = group
        if vartype == 'ordinal':
            self.dtype = PANDAS_DTYPE_MAP[self.vartype](categories=ordinal_elements, ordered=True)
        else:
            self.dtype = PANDAS_DTYPE_MAP[self.vartype]
        self.required = required

    def __repr__(self):
        return '{c}(name={n}, new_name:{m}, vartype:{v}, group={g}, dtype={d}, required={r}, ordinals={o}}'.format(
            c=self.__class__.__name__, n=self.name, m=self.new_name, v=self.vartype,
            g=self.group, d=self.dtype, r=self.required, o=self.ordinal_elements
        )


class DataReader:

    def __init__(self, config_fp):
        self._config = load_yaml_config(fp=config_fp)
        self._col_defns = self.parse_col_defs()
        self.col_groups = self._get_req_column_groups()
        self._validate_config()

    def parse_col_defs(self):
        return [ColumnDefn(**kwargs) for kwargs in self._config['column_defns']]

    def load_from_csv(self, fp, validate_col_names=True, append_vartype=False, is_test=True, pandas_csv_kwargs=None):
        # Get subset of column defns to load
        req_col_defns = [defn for defn in self._col_defns if defn.required and not (is_test and defn.group == 'target')]
        # Validate the column names
        if validate_col_names:
            source_col_names = self._get_csv_columns(fp=fp, pandas_csv_kwargs=pandas_csv_kwargs)
            req_col_names = [defn.name for defn in req_col_defns]
            DataReader._validate_source_cols(source_col_names=source_col_names, req_col_names=req_col_names)
        # Create a DataFrame from the CSV
        date_col_names = [defn.name for defn in req_col_defns if defn.dtype == 'datetime64']
        dtype_map = {defn.name: defn.dtype for defn in req_col_defns if not defn.dtype == 'datetime64'}
        df = pd.read_csv(
            filepath_or_buffer=fp,
            usecols=date_col_names + list(dtype_map.keys()),
            dtype=dtype_map,
            parse_dates=date_col_names,
            **(self._config.get('pandas_csv_kwargs') or {}),
            **(pandas_csv_kwargs or {})
        )
        # Update df column names and return it
        if append_vartype:
            col_defs_by_name = {defn.name: defn for defn in req_col_defns}
            new_col_names = [col_defs_by_name[name].new_name for name in df.columns]
            df.columns = new_col_names
        return df

    def _get_req_column_groups(self):
        groups = dict()
        groups['meta'] = [defn.new_name for defn in self._col_defns if defn.group == 'meta' and defn.required]
        groups['features'] = [defn.new_name for defn in self._col_defns if defn.group == 'feature' and defn.required]
        groups['targets'] = [defn.new_name for defn in self._col_defns if defn.group == 'target' and defn.required]
        return groups

    def _get_csv_columns(self, fp, pandas_csv_kwargs=None):
        return pd.read_csv(
            filepath_or_buffer=fp,
            **(self._config.get('pandas_csv_kwargs') or {}),
            **(pandas_csv_kwargs or {}),
            nrows=0
        ).columns

    def _validate_config(self):
        col_names = list(defn.name for defn in self._col_defns)
        if len(set(col_names)) != len(col_names):
            raise ValueError(
                'Parameter "column_defns" in config file contains duplicate columns.({duplicates})'.format(
                    duplicates=[item for item, count in collections.Counter(col_names).items() if count > 1]
                )
            )

    @staticmethod
    def _validate_source_cols(source_col_names, req_col_names):
        if len(set(source_col_names)) != len(source_col_names):
            raise ValueError(
                'Source data contains columns with duplicate names.({duplicates})'.format(
                    duplicates=[item for item, count in collections.Counter(source_col_names).items() if count > 1]
                )
            )
        if not set(req_col_names).issubset(set(source_col_names)):
            raise ValueError(
                '"{missed_cols}" in "required_col_names" were not found in the source columns'.format(
                    missed_cols=set(req_col_names).difference(set(source_col_names))
                )
            )
