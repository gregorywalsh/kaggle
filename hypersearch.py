from sklearn.model_selection import RandomizedSearchCV
from numpy import nan

def _report_on_machine(hypothesis, estimator, run_id):
    folder = './logs/hyperparameter_search/'.format(run_id=run_id)
    hypothesis.write_training_report(folder=folder, run_id=run_id, estimator=estimator)


def search_hyperparameter_space(features, target, hypothesis, scoring, cv_kwargs):
    hyperparam_optimizer = RandomizedSearchCV(
        estimator=hypothesis.estimator,
        param_distributions=hypothesis.hyperparam_dist,
        iid=False,
        return_train_score=False,
        refit=False,
        scoring=scoring,
        error_score='raise',
        **cv_kwargs
    )
    hyperparam_optimizer.fit(X=features, y=target)
    return hyperparam_optimizer.best_params_, hyperparam_optimizer.cv_results_
