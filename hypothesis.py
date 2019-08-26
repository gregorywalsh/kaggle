import numpy as np
import skorch
import skorchlogit
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
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


class AbstractHypothesis(ABC):

    def __init__(self, estimator, hyper_search_strat, hyper_search_kwargs,
                 transformer=None, additional_hyper_dists=None):
        self._estimator = estimator
        self._transformer = transformer
        self.model = self.build_model()
        additional_hyper_dists = additional_hyper_dists if additional_hyper_dists else {}
        hyper_search_kwargs['param_distributions'] = {**self._basic_hyper_dists(), **additional_hyper_dists}
        self.hyper_searcher = hyper_search_strat(estimator=self.model, **hyper_search_kwargs)
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

    def save(self, path):
        dump(self, path)

    @abstractmethod
    def preprocess(self, x_train, x_test, y_train):
        pass

    @abstractmethod
    def is_trained_stochastically(self):
        pass

    @abstractmethod
    def is_ensemble_model(self):
        pass

    @abstractmethod
    def _basic_hyper_dists(self):
        pass

    # TODO
    # @staticmethod
    # def write_training_report(folder, run_id, machine):
    #     raise NotImplementedError
    #
    # def get_feature_names(self):
    #     pass


class RandomForestHypothesis(AbstractHypothesis, ABC):

    def __init__(self, x, mode, hyper_search_strat, hyper_search_kwargs,
                 n_trees=200, preprocessor=None, y_preprocessor=None,
                 transformer=None, additional_hyper_dists=None):
        if mode == 'classification':
            estimator_class = RandomForestClassifier
        elif mode == 'regression':
            estimator_class = RandomForestRegressor
        else:
            raise ValueError('"mode" must be one of {"classification", "regression"}')
        super().__init__(
            estimator=estimator_class(n_jobs=1, bootstrap=True, n_estimators=n_trees),
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            transformer=transformer,
            additional_hyper_dists=additional_hyper_dists
        )
        self.x = x

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    # TODO fix this
    # def _basic_hyper_dists(self):
    #     raise NotImplementedError
    #     x = self.preprocess(df_train, df_test) if self._preprocessor else df_train, df_test
    #     steps = []
    #     if self._cv_transformer:
    #         steps.append(('cv_transformer', self._cv_transformer))
    #     pipeline = Pipeline(steps=steps)
    #     feature_count = get_feature_count(dataset=x, pipeline=pipeline)
    #     sample_count = x.shape[0]
    #     dist = {
    #         'estimator__max_depth': binom(sample_count - 1, (np.log(sample_count**2) / np.log(2)) / sample_count, 1),
    #         'estimator__criterion': ["gini", "entropy"],
    #         'estimator__max_features': binom(feature_count - 1, 1/(feature_count**0.5 + 1), 1),
    #         'estimator__min_samples_split': loguniform(a=1/sample_count, b=0.49, base=10)
    #     }
    #     print(dist)
    #     return dist


class LogisticRegressionHypothesis(AbstractHypothesis, ABC):

    def __init__(self, hyper_search_strat, hyper_search_kwargs, penalty=None,
                 solver=None, tol=None, max_iter=None, preprocessor=None,
                 transformer=None, additional_hyper_dists=None):
        self.solver = solver
        self.penalty = penalty
        super().__init__(
            estimator=LogisticRegression(penalty=penalty, solver=solver, tol=tol, max_iter=max_iter),
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            transformer=transformer,
            additional_hyper_dists=additional_hyper_dists
        )

    def preprocess(self, x_train, x_test, y_train):
        raise NotImplementedError

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return self.solver in ('sag', 'saga', )

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    # TODO fix this
    # def _basic_hyper_dists(self):
    #     dist = {
    #             'estimator__c': [10**x for x in range(-3, 5)],
    #     }
    #     if self.penalty == 'elasticnet':
    #         dist['estimator__l1_ratio'] = uniform(0, 1)
    #     return dist


class LightGBMClassifierHypothesis(AbstractHypothesis, ABC):

    def __init__(self, x, hyper_search_strat, hyper_search_kwargs, n_trees=512,
                 preprocessor=None, transformer=None, additional_hyper_dists=None):
        super().__init__(
            estimator=LGBMClassifier(n_estimators=n_trees),
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            transformer=transformer,
            additional_hyper_dists=additional_hyper_dists
        )
        self.x = x

    def preprocess(self, x_train, x_test, y_train):
        raise NotImplementedError

    def is_trained_stochastically(self):
        # RF training is stochastic due to bagging of training data and features per tree
        return True

    def is_ensemble_model(self):
        # RF contains ensemble of bootstrapped trees AKA bagging
        return True

    # TODO fix this
    # def _basic_hyper_dists(self):
    #     sample_count = self.x.shape[0]
    #     dist = {
    #         'estimator__min_data_in_leaf': randint(1, 50),
    #         'estimator__min_child_samples': [None]
    #     }
    #     return dist


class NNHypothesis(AbstractHypothesis, ABC):

    def __init__(self, mode, module, use_gpu, checkpoint_dir, hyper_search_strat, hyper_search_kwargs,
                 net_kwargs=None, transformer=None, additional_hyper_dists=None):
        if mode == 'classification':
            estimator_class = skorchlogit.LogitNeuralNetClassifier
        elif mode == 'regression':
            estimator_class = skorch.NeuralNetRegressor
        else:
            raise ValueError('"mode" must be one of {"classification", "regression"}')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        checkpoint = skorch.callbacks.Checkpoint(
            monitor='valid_loss_best', f_params='nn_params.pt', dirname=checkpoint_dir
        )
        early_stopping = skorch.callbacks.EarlyStopping(
            monitor='valid_loss',
            patience=1,
            threshold=0,
            threshold_mode='rel',
            lower_is_better=True,
        )
        net_kwargs = {"module__" + key: val for key, val in net_kwargs.items()}
        estimator = estimator_class(
            module=module,
            device=self.device,
            callbacks=[('early_stopping', early_stopping), ('checkpoint', checkpoint)],
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=5,
            iterator_train__shuffle=True,
            iterator_train__drop_last=True,
            iterator_valid__drop_last=False,
            iterator_train__batch_size=128,
            iterator_valid__batch_size=-1,  # use all examples
            verbose=1,
            **net_kwargs,
        )
        super().__init__(
            estimator=estimator,
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            transformer=transformer,
            additional_hyper_dists=additional_hyper_dists
        )

    def is_trained_stochastically(self):
        # RF training is stochastic due to SGD and random initialization of model
        return True

    def is_ensemble_model(self):
        # NN are a single model
        return False

    def _basic_hyper_dists(self):
        return {}
