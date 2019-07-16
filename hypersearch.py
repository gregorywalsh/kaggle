import csv
from operator import itemgetter


def search_hyperparam_space(hypothesis):
    searcher = hypothesis.hyper_searcher
    target = hypothesis.dataset.target
    y = target.values.ravel() if len(target.columns) == 1 else target.values
    searcher.fit(X=hypothesis.dataset.features, y=y)
    return searcher.best_estimator_, searcher.best_params_, searcher.cv_results_


def process_cv_results(run_id, hypothesis_name, cv_results, cv_reporting_keys):
    cv_key_results = list(zip(*[cv_results[key] for key in cv_reporting_keys]))
    all_splits = list(zip(*[cv_results[key] for key in cv_results.keys() if key.startswith('split')]))
    all_min_maxes = [(min(splits), max(splits)) for splits in all_splits]
    # TODO - ADD NUM FOLDS AND NUM HYP_SAMP
    all_results = [[run_id, hypothesis_name, *data, *min_max] for data, min_max in zip(cv_key_results, all_min_maxes)]
    titles = ['run_id', 'run_id'] + cv_reporting_keys + ['min_score', 'max_score']
    return [titles] + sorted(all_results, key=itemgetter(2))


def save_cv_results(path, processed_cv_results, top_n=-1):
    top_n_results = processed_cv_results[1:top_n]
    with open(file=path, mode='a+') as f:
        f_writer = csv.writer(f, dialect='excel')
        f_writer.writerows(top_n_results)
    return


def save_model_repr(path, model):
    with open(file=path, mode='a+') as f:
        f.write(model.__repr__(float('inf')))  # Get all chars with float('inf')
