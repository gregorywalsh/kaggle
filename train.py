from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.base import clone
import numpy as np
import pandas as pd

def train_machine(X, y, conduit, rand_search_kwargs, splits):
    final_machine = RandomizedSearchCV(
            estimator=conduit.pipeline,
            param_distributions=conduit.hyperparam_dist,
            iid=False,
            cv=splits,
            return_train_score=False,
            **rand_search_kwargs
        )
    final_machine.fit(X, y)

    return final_machine.best_estimator_
