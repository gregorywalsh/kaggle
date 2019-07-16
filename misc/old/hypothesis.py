import numpy as np
from dataclasses import dataclass
from warnings import warn
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.base import clone as clone_estimator
from sklearn.pipeline import Pipeline


@dataclass
class Hypothesis:
    pre_cv_pipeline: Pipeline
    cv_pipeline: Pipeline
    hyperparam_dist: dict


def _skl_rand_forest_dist(features, pipeline):

    def get_feature_count(features, pipeline):
        if pipeline:
            pipeline_clone = clone_estimator(pipeline)
            feature_count = pipeline_clone.fit_transform(features).shape[1]
        else:
            feature_count = features.shape[1]
        return feature_count

    feature_count = get_feature_count(features, pipeline)
    dist = {
        'classifier__max_depth': [2 ** x for x in range(0, 6)] + [None],
        'classifier__criterion': ["gini", "entropy"],
        'classifier__max_features': randint(int(feature_count ** 0.4), int(feature_count ** 0.6)),
        'classifier__min_samples_split': uniform(0, 0.5)
    }
    return dist


def get_hyperparam_dist(features, pre_cv_pipeline, cv_pipeline):

    steps = []
    if pre_cv_pipeline:
        steps.append(('pre_cv_pipeline', pre_cv_pipeline))
    else:
        warn("No pre CV pipeline given")
    if 'preprocessor' in cv_pipeline.named_steps:
        steps.append(('preprocessor', Pipeline(cv_pipeline.named_steps['preprocessor'])))
    else:
        warn("No 'preprocessing' step found in cv_pipeline")
    if 'estimator' in cv_pipeline.named_steps:
        estimator = Pipeline(cv_pipeline.named_steps['estimator'])
    else:
        raise ValueError("No 'estimator' step found in pipeline")
    pipeline = Pipeline(steps=steps)

    if isinstance(estimator, RandomForestClassifier):
        hyperparam_dist = _skl_rand_forest_dist(features, pipeline)
    else:
        raise ValueError("Cannot create distribution for estimator '{}'".format(estimator.__class__.__name__))

    return hyperparam_dist