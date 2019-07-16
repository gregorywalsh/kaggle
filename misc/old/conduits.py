import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone as clone_estimator


class AbstractConduit(Pipeline):

    def __init__(self, dataset, base_classifier, fitted_transformer=None):
        self._dataset = dataset
        self._base_classifier = base_classifier
        self._fitted_transformer = fitted_transformer
        self._dimensionality_transformer_groups = self._get_dimensionality_transformer_groups()
        super().__init__(steps=self._build_steps())
        self._hyperparam_dist = self._build_hyperparam_dist()

    def _build_steps(self):
        pipeline_steps = []
        if self._fitted_transformer:
            pipeline_steps.append(('fitted_transformer', FunctionTransformer(func=self._fitted_transformer)))
        dimensionality_transformer = self._get_dimensionality_transformer()
        if dimensionality_transformer:
            pipeline_steps.append(('dimensionality_transformer', dimensionality_transformer))
        pipeline_steps.append(('classifier', self._base_classifier))
        return pipeline_steps

    def _get_dimensionality_transformer_groups(self):
        groups = {}
        for col in self._dataset.features.columns:
            _, vartype = col.split('|')
            preprocessing_grp = self.__class__.DIMENSIONALITY_MAP.get(vartype)
            if preprocessing_grp in groups:
                groups[preprocessing_grp].append(col)
            else:
                groups[preprocessing_grp] = [col]
        return groups

    @staticmethod
    def write_training_report(folder, run_id, machine):
        raise NotImplementedError

    def _get_dimensionality_transformer(self):
        raise NotImplementedError

    def _build_hyperparam_dist(self):
        raise NotImplementedError

    def get_feature_names(self):
        raise NotImplementedError


class SKLRandomForestConduit(AbstractConduit):

    DIMENSIONALITY_MAP = {
        'boolean': 'categorical',
        'categorical': 'categorical',
        'continuous': 'numerical',
        'discrete': 'numerical',
    }

    def __init__(self, dataset, fitted_transformer, n_trees=100):
        super().__init__(
            dataset=dataset,
            base_classifier=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            fitted_transformer=fitted_transformer
        )

    def _get_dimensionality_transformer(self):

        def one_hot_encode(dataset):
            pass

        col_transformers = []
        if 'categorical' in self._dimensionality_transformer_groups:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', dtype='float32'))
            ])
            col_transformers.append(('categorical', cat_transformer, self._dimensionality_transformer_groups['categorical']))
        return None
        return ColumnTransformer(transformers=col_transformers, remainder='passthrough') if col_transformers else None

    def _build_hyperparam_dist(self):
        def _get_transformed_feature_count(conduit):
            clone_steps = []
            if 'feature_eng' in self.named_steps:
                clone_steps.append('clone_estimator(conduit.named_steps['feature_eng']))
            if 'dimensionality_transformer' in self.named_steps:
                clone_steps.append(('estimator', clone_estimator(conduit.named_steps['dimensionality_transformer'])))
            if clone_steps:
                feature_count = Pipeline(steps=(clone_steps).fit_transform(conduit.dataset.features).shape[1]
            else:
                feature_count = conduit.dataset.df_features.shape[1]
            return feature_count

        num_features = _get_transformed_feature_count(self)
        dist = {
            'classifier__max_depth': [2**x for x in range(0, 6)] + [None],
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_features': randint(1, (num_features ** 0.5) + 3),
            'classifier__min_samples_split': uniform(0, 0.5)
        }
        return dist

    # def get_feature_names(self):
    #     transformers = self.transformer.transformers_
    #     col_out_names = []
    #     for trans_name, pipe, col_ids in transformers:
    #         if trans_name == 'numerical':
    #                 col_out_names += col_ids
    #         # TODO implement getting categorical names for pandas backend
    #         # if trans_name == 'cat':
    #         #     one_hot_encoder = pipe.named_steps.get('onehot', None)
    #         #     if one_hot_encoder:
    #         #         raw_cat_names_out = one_hot_encoder.get_feature_names()
    #         #         for raw_cat_name in raw_cat_names_out:
    #         #             column_index = int(raw_cat_name.split('_')[0][1:])
    #         #             column_child_id = str(raw_cat_name.split('_')[1])
    #         #             col_out_names.append('{}_{}'.format(col_ids[column_index], column_child_id))
    #     return col_out_names

    @staticmethod
    def write_training_report(folder, run_id, pipeline):

        def feature_importances(pipeline):
            clf = pipeline.named_steps['classifier']
            importances = clf.feature_importances_
            out_col_names = SKLRandomForestConduit.get_feature_names()
            indices = np.argsort(importances)[::-1]
            output = 'Feature ranking:\n'
            for f in range(len(out_col_names)):
                output += '{}. feature {} ({})\n'.format(f + 1, out_col_names[indices[f]], importances[indices[f]])
            return output

        with open(file='{folder}/{run_id}/random_forest_report.txt'.format(folder=folder, run_id=run_id)) as f:
            f.write(
                "{cv_results}".format(
                    cv_results=feature_importances(pipeline)
                )
            )


class LogRegConduit(AbstractConduit):

    def __init__(self, dataset, penalty, dual, preproc_feature_groups, fitted_transformer=None):
        self.penalty = penalty
        super().__init__(
            dataset=dataset,
            base_classifier=LogisticRegression(
                n_jobs=1,
                dual=dual,
                tol=1e-5,
                max_iter=10000,
                # solver={'l2': 'sag', 'l1': 'saga'}[penalty],
                solver={'l2': 'lbfgs', 'l1': 'liblinear'}[penalty]
            ),
            preproc_feature_groups=preproc_feature_groups,
            fitted_transformer=fitted_transformer
        )

    def _get_dimensionality_transformer(self):
        column_transformers = []
        if 'numerical' in self._dimensionality_transformer_groups:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('standardise', StandardScaler(with_mean=True, with_std=True))
            ])
            column_transformers.append(('numerical', num_transformer, self._dimensionality_transformer_groups['numerical']))

        if 'categorical' in self._dimensionality_transformer_groups:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', dtype='float32')),
                ('standardise', StandardScaler(with_mean=True, with_std=True))
            ])
            column_transformers.append(('categorical', cat_transformer, self._dimensionality_transformer_groups['categorical']))

        return ColumnTransformer(transformers=column_transformers) if column_transformers else None

    def _build_hyperparam_dist(self):
        dist = {
            'classifier__C': [4**x for x in range(-5, 6)],
            }
        dist['classifier__penalty'] = [self.penalty]
        return dist

    def get_feature_names(self):
        transformers = self._fitted_transformer.transformers_
        col_out_names = []
        for trans_name, pipe, col_ids in transformers:
            if trans_name == 'numerical':
                    col_out_names += col_ids
            # TODO implement getting categorical names for both numpy and pandas backend
            # if trans_name == 'cat':
            #     one_hot_encoder = pipe.named_steps.get('onehot', None)
            #     if one_hot_encoder:
            #         raw_cat_names_out = one_hot_encoder.get_feature_names()
            #         for raw_cat_name in raw_cat_names_out:
            #             column_index = int(raw_cat_name.split('_')[0][1:])
            #             column_child_id = str(raw_cat_name.split('_')[1])
            #             col_out_names.append('{}_{}'.format(col_ids[column_index], column_child_id))

        return col_out_names

    @staticmethod
    def write_training_report(folder, run_id, machine):
        # TODO - Write out feature importances and scoring information to a file
        raise NotImplementedError