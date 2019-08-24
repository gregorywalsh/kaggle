import numpy as np
import skorch
import torch
import torch.nn as nn

from distributions import loguniform
from joblib import load, dump
from lightgbm.sklearn import LGBMClassifier
from scipy.stats import uniform, binom, randint
from sklearn.base import clone as clone_estimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skorch.helper import predefined_split


def load_hypothesis(path):
    return load(path)


def get_feature_count(dataset, pipeline):
    if pipeline:
        pipeline_clone = clone_estimator(pipeline)
        feature_count = pipeline_clone.fit_transform(X=dataset.features, y=dataset.target).shape[1]
    else:
        feature_count = dataset.features.shape[1]
    return feature_count


class AbstractHypothesis:

    def __init__(self, estimator, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None):
        self._estimator = estimator
        self._x_preprocessor = x_preprocessor
        self._y_preprocessor = y_preprocessor
        self._cv_transformer = cv_transformer
        self.model = self.build_model()
        hyper_searcher_kwargs['param_distributions'] = {**self._basic_hyper_dists(), **additional_hyper_dists}
        self.hyper_searcher = hyper_searcher_strategy(estimator=self.model, **hyper_searcher_kwargs)
        self.training_is_stochastic = self.is_trained_stochastically()
        self.best_model = None
        self.best_hyperparams = None
        self.hyper_search_results = None

    def build_model(self):
        steps = []
        if self._cv_transformer:
            steps.append(('transformer', self._cv_transformer))
        steps.append(('estimator', self._estimator))
        return Pipeline(steps=steps)

    def preprocess_x(self, x_train, x_test):
        return self._x_preprocessor(x_train=x_train, x_test=x_test) if self._x_preprocessor else (x_train, x_test)

    def preprocess_y(self, y_train):
        return self._y_preprocessor(y_train=y_train) if self._y_preprocessor else y_train

    def is_trained_stochastically(self):
        raise NotImplementedError

    def is_ensemble_model(self):
        raise NotImplementedError

    def save_hypothesis(self, path):
        dump(self, path)

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

    def __init__(self, x, estimator, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None):
        super().__init__(
            estimator=estimator,
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            x_preprocessor=x_preprocessor,
            y_preprocessor=y_preprocessor,
            cv_transformer=cv_transformer,
            additional_hyper_dists=additional_hyper_dists
        )
        self.x = x

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _basic_hyper_dists(self):

        x = self._x_preprocessor(self.x) if self._x_preprocessor else self.x
        steps = []
        if self._cv_transformer:
            steps.append(('cv_transformer', self._cv_transformer))
        pipeline = Pipeline(steps=steps)
        feature_count = get_feature_count(dataset=x, pipeline=pipeline)
        sample_count = x.shape[0]
        dist = {
            'estimator__max_depth': binom(sample_count - 1, (np.log(sample_count**2) / np.log(2)) / sample_count, 1),
            'estimator__criterion': ["gini", "entropy"],
            'estimator__max_features': binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1),
            'estimator__min_samples_split': loguniform(a=1/sample_count, b=0.49, base=10)
        }
        print(dist)
        return dist


class RandomForrestClassifierHypothesis(RandomForestHypothesis):

    def __init__(self, x, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None, n_trees=200):
        super().__init__(
            x=x,
            estimator=RandomForestClassifier(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            x_preprocessor=x_preprocessor,
            y_preprocessor=y_preprocessor,
            cv_transformer=cv_transformer,
            additional_hyper_dists=additional_hyper_dists
        )


class RandomForrestRegressorHypothesis(RandomForestHypothesis):

    def __init__(self, x, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None, n_trees=200):
        super().__init__(
            x=x,
            estimator=RandomForestRegressor(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            x_preprocessor=x_preprocessor,
            y_preprocessor=y_preprocessor,
            cv_transformer=cv_transformer,
            additional_hyper_dists=additional_hyper_dists
        )


class LogisticRegressionHypothesis(AbstractHypothesis):

    def __init__(self, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None,
                 penalty=None, solver=None, tol=None, max_iter=None):
        self.solver = solver
        self.penalty = penalty
        super().__init__(
            estimator=LogisticRegression(penalty=penalty, solver=solver, tol=tol, max_iter=max_iter),
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            x_preprocessor=x_preprocessor,
            y_preprocessor=y_preprocessor,
            cv_transformer=cv_transformer,
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


class LightGBMClassifierHypothesis(AbstractHypothesis):

    def __init__(self, x, hyper_searcher_strategy, hyper_searcher_kwargs, x_preprocessor=None,
                 y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None, n_trees=512):
        super().__init__(
            estimator=LGBMClassifier(n_estimators=n_trees),
            hyper_searcher_strategy=hyper_searcher_strategy,
            hyper_searcher_kwargs=hyper_searcher_kwargs,
            x_preprocessor=x_preprocessor,
            y_preprocessor=y_preprocessor,
            cv_transformer=cv_transformer,
            additional_hyper_dists=additional_hyper_dists
        )
        self.x = x

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    def _basic_hyper_dists(self):
        sample_count = self.x.shape[0]
        dist = {
            'estimator__min_data_in_leaf': randint(1, 50),
            'estimator__min_child_samples': [None]
        }
        return dist


class NNHypothesis(AbstractHypothesis):

    def __init__(self, module, val_dataset, use_gpu, hyper_searcher_strategy, hyper_searcher_kwargs,
                 x_preprocessor=None, y_preprocessor=None, cv_transformer=None, additional_hyper_dists=None):

        checkpoint = skorch.callbacks.Checkpoint(
            monitor='valid_loss_best',
            f_params='nn_params.pt',
            dirname='challenges/spooky_author_identification/checkpoints'
        )
        early_stopping = skorch.callbacks.EarlyStopping(
            monitor='valid_loss',
            patience=1,
            threshold=0,
            threshold_mode='rel',
            lower_is_better=True,
        )
        estimator = skorch.NeuralNetClassifier(
            module=module,
            device=torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"),
            callbacks=[('early_stopping', early_stopping), ('checkpoint', checkpoint)],
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=predefined_split(val_dataset),
            iterator_train__shuffle=True,
            iterator_train__drop_last=True,
            iterator_valid__drop_last=False,
            iterator_train__batch_size=128,
            iterator_valid__batch_size=-1,  # use all examples
            verbose=1
        )
        super().__init__(estimator=estimator, hyper_searcher_strategy=hyper_searcher_strategy,
                         hyper_searcher_kwargs=hyper_searcher_kwargs, x_preprocessor=x_preprocessor,
                         y_preprocessor=y_preprocessor, cv_transformer=cv_transformer,
                         additional_hyper_dists=additional_hyper_dists)

    def is_trained_stochastically(self):
        # RF training is stochastic due to SGD and random initialization of model
        return True

    def is_ensemble_model(self):
        # NN are a single model
        return False

    def _basic_hyper_dists(self):
        return {}
