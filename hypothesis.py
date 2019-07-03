import numpy as np
from warnings import warn
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone as clone_estimator


class AbstractHypothesis:

    @staticmethod
    def get_categorical_embedding_pipeline(test_feature, train_feature, missing_values=('',)):
        raise NotImplementedError

    def __init__(self, features, base_classifier, cv_transformer=None, best_hyperparams=None):
        self._features = features
        self._base_classifier = base_classifier
        self._cv_transformer = cv_transformer
        self.estimator = Pipeline(steps=[
            ('cv_transforms', cv_transformer),
            ('estimator', base_classifier)
        ])
        self.training_is_stochastic = self.get_training_is_stochastic_flag()
        self.hyperparam_dist = self._build_hyperparam_dist()
        self.best_hyperparams = best_hyperparams

    def get_training_is_stochastic_flag(self):
        raise NotImplementedError

    def _build_hyperparam_dist(self):
        pass

    # TODO
    # @staticmethod
    # def write_training_report(folder, run_id, machine):
    #     raise NotImplementedError
    #
    # def get_feature_names(self):
    #     pass


class RandomForestHypothesis(AbstractHypothesis):

    DIMENSIONALITY_MAP = {
        'boolean': 'categorical',
        'categorical': 'categorical',
        'continuous': 'numerical',
        'discrete': 'numerical',
    }

    @staticmethod
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

    def __init__(self, base_classifier, features, cv_transformer, n_trees=100):
        super().__init__(
            features=features,
            base_classifier=base_classifier,
            cv_transformer=cv_transformer
        )

    def get_training_is_stochastic_flag(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def _build_hyperparam_dist(self):

        def get_feature_count(features, pipeline):
            if pipeline:
                pipeline_clone = clone_estimator(pipeline)
                feature_count = pipeline_clone.fit_transform(features).shape[1]
            else:
                feature_count = features.shape[1]
            return feature_count

        steps = []
        if self._cv_transformer:
            steps.append(('cv_transforms', self._cv_transformer))
        else:
            warn("No CV transforms given")
        pipeline = Pipeline(steps=steps)
        feature_count = get_feature_count(features=self._features, pipeline=pipeline)
        dist = {
            'estimator__max_depth': [2 ** x for x in range(0, 6)] + [None],
            'estimator__criterion': ["gini", "entropy"],
            'estimator__max_features': randint(max(1, int(feature_count ** 0.4)), max(2, int(feature_count ** 0.6))),
            'estimator__min_samples_split': uniform(0, 0.5)
        }
        return dist


class RandomForrestClassifierHypothesis(RandomForestHypothesis):
    def __init__(self, features, cv_transformer, n_trees=100):
        super().__init__(
            features=features,
            base_classifier=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            cv_transformer=cv_transformer
        )


class RandomForrestRegressorHypothesis(RandomForestHypothesis):
    def __init__(self, features, cv_transformer, n_trees=100):
        super().__init__(
            features=features,
            base_classifier=RandomForestRegressor(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            cv_transformer=cv_transformer
        )
