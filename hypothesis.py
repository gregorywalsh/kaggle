import numpy as np
from warnings import warn
from scipy.stats import randint, uniform, binom
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone as clone_estimator


class LogUniform:
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform_dist = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, uniform_dist.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform_dist.rvs(size=size, random_state=random_state))


class AbstractHypothesis:

    def __init__(self, dataset, classifier, pre_transformer=None, best_hyperparams=None):
        self._dataset = dataset
        self._classifier = classifier
        self._pre_transformer = pre_transformer
        self.estimator = Pipeline(steps=[
            ('pre_transformer', pre_transformer),
            ('estimator', classifier)
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

    def __init__(self, classifier, dataset, pre_transformer):
        super().__init__(
            dataset=dataset,
            classifier=classifier,
            pre_transformer=pre_transformer
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _build_hyperparam_dist(self):

        def get_feature_count(dataset, pipeline):
            if pipeline:
                pipeline_clone = clone_estimator(pipeline)
                feature_count = pipeline_clone.fit_transform(X=dataset.features, y=dataset.target).shape[1]
            else:
                feature_count = dataset.features.shape[1]
            return feature_count

        steps = []
        if self._pre_transformer:
            steps.append(('pre_transformer', self._pre_transformer))
        pipeline = Pipeline(steps=steps)
        feature_count = get_feature_count(dataset=self._dataset, pipeline=pipeline)
        dist = {
            'estimator__max_depth': [2**x for x in range(0, 8)] + [None],
            # 'estimator__max_depth': binom(data_count - 1, (np.log(data_count**2) / np.log(2)) / data_count, 1),
            'estimator__criterion': ["gini", "entropy"],
            'estimator__max_features': binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1),
            'estimator__min_samples_split': uniform(0, 0.2)
        }
        return dist


class RandomForrestClassifierHypothesis(RandomForestHypothesis):
    def __init__(self, dataset, pre_transformer, n_trees=100):
        super().__init__(
            dataset=dataset,
            classifier=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            pre_transformer=pre_transformer
        )


class RandomForrestRegressorHypothesis(RandomForestHypothesis):
    def __init__(self, dataset, pre_transformer, n_trees=100):
        super().__init__(
            dataset=dataset,
            classifier=RandomForestRegressor(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            pre_transformer=pre_transformer
        )


class LogisticRegressionHypothesis(AbstractHypothesis):

    def __init__(self, dataset, pre_transformer, penalty, solver, tol=None, max_iter=None):
        self.solver = solver
        self.penalty = penalty
        super().__init__(
            dataset=dataset,
            classifier=LogisticRegression(penalty=penalty, solver=solver, tol=tol, max_iter=max_iter),
            pre_transformer=pre_transformer
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return self.solver in ('sag', 'saga', )

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _build_hyperparam_dist(self):
        dist = {
                'estimator__c': [10**x for x in range(-3, 5)],
        }
        if self.penalty == 'elasticnet':
            dist['estimator__l1_ratio'] = uniform(0, 1)
        return dist

a = np.log