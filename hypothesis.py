import numpy as np
from warnings import warn
from scipy.stats import randint, uniform, binom, geom
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone as clone_estimator


class AbstractHypothesis:

    def __init__(self, features, base_classifier, pre_transformer=None, best_hyperparams=None):
        self._features = features
        self._base_classifier = base_classifier
        self._pre_transformer = pre_transformer
        self.estimator = Pipeline(steps=[
            ('cv_transforms', pre_transformer),
            ('estimator', base_classifier)
        ])
        self.training_is_stochastic = self.is_trained_stochastically()
        self.hyperparam_dist = self._build_hyperparam_dist()
        self.best_hyperparams = best_hyperparams

    def is_trained_stochastically(self):
        raise NotImplementedError

    def is_ensemble_model(self):
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

    def __init__(self, base_classifier, features, pre_transformer):
        super().__init__(
            features=features,
            base_classifier=base_classifier,
            pre_transformer=pre_transformer
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
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
        if self._pre_transformer:
            steps.append(('cv_transforms', self._pre_transformer))
        else:
            warn("No CV transforms given")
        pipeline = Pipeline(steps=steps)
        feature_count = get_feature_count(features=self._features, pipeline=pipeline)
        data_count = self._features.shape[0]
        dist = {
            'estimator__max_depth': [2**x for x in range(0, 8)] + [None],
            # 'estimator__max_depth': binom(data_count - 1, (np.log(data_count**2) / np.log(2)) / data_count, 1),
            'estimator__criterion': ["gini", "entropy"],
            'estimator__max_features': binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1),
            'estimator__min_samples_split': uniform(0, 0.2)
        }
        return dist


class RandomForrestClassifierHypothesis(RandomForestHypothesis):
    def __init__(self, features, pre_transformer, n_trees=100):
        super().__init__(
            features=features,
            base_classifier=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            pre_transformer=pre_transformer
        )


class RandomForrestRegressorHypothesis(RandomForestHypothesis):
    def __init__(self, features, pre_transformer, n_trees=100):
        super().__init__(
            features=features,
            base_classifier=RandomForestRegressor(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            pre_transformer=pre_transformer
        )


class LogisticRegressionHypothesis(AbstractHypothesis):

    DIMENSIONALITY_MAP = {
        'boolean': 'categorical',
        'categorical': 'categorical',
        'continuous': 'numerical',
        'discrete': 'numerical',
    }

    def __init__(self, pre_transformer):
        super().__init__(
            features=features,
            base_classifier=base_classifier,
            pre_transformer=pre_transformer
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
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
        if self._pre_transformer:
            steps.append(('cv_transforms', self._pre_transformer))
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
