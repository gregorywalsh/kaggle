import numpy as np
from scipy.stats import uniform, binom
from distributions import loguniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone as clone_estimator
from joblib import load, dump


def save_hypothesis(hypothesis, path):
    dump(hypothesis, path)


def load_hypothesis(path):
    return load(path)


class AbstractHypothesis:

    def __init__(self, dataset, estimator, hyper_searcher_strategy, hyper_searcher_kwargs,
                 transformer=None, additional_hyper_dists=None):
        self.dataset = dataset
        self._estimator = estimator
        self._transformer = transformer
        self.model = self.build_model()
        hyper_searcher_kwargs['param_distributions'] = {**self._basic_hyper_dists(), **additional_hyper_dists}
        self.hyper_searcher = hyper_searcher_strategy(estimator=self.model, **hyper_searcher_kwargs)
        self.training_is_stochastic = self.is_trained_stochastically()
        self.best_model = None
        self.best_hyperparams = None
        self.hyper_search_results = None

    def build_model(self):
        steps = []
        if self._transformer:
            steps.append(('transformer', self._transformer))
        steps.append(('estimator', self._estimator))
        return Pipeline(steps=steps)

    def is_trained_stochastically(self):
        raise NotImplementedError

    def is_ensemble_model(self):
        raise NotImplementedError

    def _basic_hyper_dists(self):
        raise NotImplementedError

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['dataset']
        return state

    # TODO
    # @staticmethod
    # def write_training_report(folder, run_id, machine):
    #     raise NotImplementedError
    #
    # def get_feature_names(self):
    #     pass


class RandomForestHypothesis(AbstractHypothesis):

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _basic_hyper_dists(self):

        def get_feature_count(dataset, pipeline):
            if pipeline:
                pipeline_clone = clone_estimator(pipeline)
                feature_count = pipeline_clone.fit_transform(X=dataset.features, y=dataset.target).shape[1]
            else:
                feature_count = dataset.features.shape[1]
            return feature_count

        steps = []
        if self._transformer:
            steps.append(('pre_transformer', self._transformer))
        pipeline = Pipeline(steps=steps)
        feature_count = get_feature_count(dataset=self.dataset, pipeline=pipeline)
        sample_count = self.dataset.features.shape[0]
        dist = {
            'estimator__max_depth': binom(sample_count - 1, (np.log(sample_count**2) / np.log(2)) / sample_count, 1),
            'estimator__criterion': ["gini", "entropy"],
            'estimator__max_features': binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1),
            'estimator__min_samples_split': loguniform(a=1/sample_count, b=0.49, base=10)
        }
        print(dist)
        return dist


class RandomForrestClassifierHypothesis(RandomForestHypothesis):
    def __init__(self, dataset, hyper_searcher_strategy, hyper_searcher_kwargs, transformer=None,
                 additional_hyper_dists=None, n_trees=200):
        super().__init__(
            dataset=dataset,
            estimator=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            transformer=transformer,
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            additional_hyper_dists=additional_hyper_dists
        )


class RandomForrestRegressorHypothesis(RandomForestHypothesis):
    def __init__(self, dataset, hyper_searcher_strategy, hyper_searcher_kwargs, transformer=None,
                 additional_hyper_dists=None, n_trees=200):
        super().__init__(
            dataset=dataset,
            estimator=RandomForestRegressor(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            transformer=transformer,
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            additional_hyper_dists=additional_hyper_dists
        )


class LogisticRegressionHypothesis(AbstractHypothesis):

    def __init__(self, dataset, hyper_searcher_kwargs, hyper_searcher_strategy, transformer=None,
                 additional_hyper_dists=None, penalty=None, solver=None, tol=None, max_iter=None):
        self.solver = solver
        self.penalty = penalty
        super().__init__(
            dataset=dataset,
            estimator=LogisticRegression(penalty=penalty, solver=solver, tol=tol, max_iter=max_iter),
            transformer=transformer,
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            additional_hyper_dists=additional_hyper_dists
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return self.solver in ('sag', 'saga', )

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _basic_hyper_dists(self):
        dist = {
                'estimator__c': [10**x for x in range(-3, 5)],
        }
        if self.penalty == 'elasticnet':
            dist['estimator__l1_ratio'] = uniform(0, 1)
        return dist
