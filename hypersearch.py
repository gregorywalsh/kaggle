from sklearn.model_selection import RandomizedSearchCV
import csv


def search_hyperparam_space(features, target, hypothesis, scoring, cv_kwargs):
    hyperparam_optimizer = RandomizedSearchCV(
        estimator=hypothesis.model,
        param_distributions=hypothesis.hyperparam_dist,
        iid=True,
        return_train_score=False,
        refit=False,
        scoring=scoring,
        error_score='raise',
        **cv_kwargs
    )
    hyperparam_optimizer.fit(X=features, y=target)
    return hyperparam_optimizer.best_params_, hyperparam_optimizer.cv_results_


def save_cv_results(path, run_id, hypothesis_name, cv_results, cv_reporting_keys):
    cv_key_results = list(zip(*[cv_results[key] for key in cv_reporting_keys]))
    all_splits = list(zip(*[cv_results[key] for key in cv_results.keys() if key.startswith('split')]))
    all_min_maxes = [(min(splits), max(splits)) for splits in all_splits]
    all_results = [[run_id, hypothesis_name, *data, *min_max] for data, min_max in zip(cv_key_results, all_min_maxes)]
    with open(file=path, mode='a+') as f:
        f_writer = csv.writer(f, dialect='excel')
        f_writer.writerows(sorted(all_results))


def save_model_repr(path, model):
    with open(file=path, mode='a+') as f:
        f.write(model.__repr__(float('inf')))  # Get all chars with float('inf')
