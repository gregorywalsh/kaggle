import importlib
from time import time
from argparse import ArgumentParser
from dataset import Dataset
from hypotheses import save_hypothesis
from util import print_title, process_cv_results, save_cv_results


# STATIC VARIABLES
# ======================================================================================================================
REPORTING_KEYS = ['run_id', 'hypothesis_name', 'cv_folds', 'cv_repeats', 'num_hyp_samples',
                  'rank_test_score', 'mean_test_score', 'std_test_score', 'min_score',
                  'max_score', 'mean_fit_time', 'params']


# PARSE ARGS
# ======================================================================================================================
parser = ArgumentParser(description='Train and evaluate bunch of models')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-H', '--num_hyp_samples', type=int, default=0,
                    help='The number of hyperparam samples per hypothesis')
parser.add_argument('-K', '--cv_folds', type=int, default=10,
                    help='The number of CV folds')
parser.add_argument('-P', '--cv_repeats', type=int, default=2,
                    help='The number of CV repeats')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
parser.add_argument('-D', '--num_dispatch', type=int, default=None,
                    help='The number of jobs to dispatch initially')
parser.add_argument('-R', '--num_rows', type=int, default=None,
                    help='The number of rows or fraction in interval [0,1] to use in training')
parser.add_argument('-T', '--report_n', type=int, default=None,
                    help='The number of models on which to report training performance')
args = parser.parse_args()


# MAIN
# ======================================================================================================================
print_title('BEGINNING RUN')
print('Challenge {c}\n'.format(c=args.challenge))


print_title('LOADING TEST AND TRAIN DATA')
# TODO: Add some validation for test vs training col_defns (or at the fit/predict stage on conduit)
train = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/train.csv'.format(args.challenge),
    is_test=False,
    num_rows=args.num_rows,
    verbose=True,
    always_validate=True
)
print('{r} training rows loaded\n'.format(r=train.features.shape[0]))


print_title('MUNGING TRAINING DATA')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
train.features = challenge.munge_features(train.features)
print('done\n')


print_title('BUILDING HYPOTHESES')
hyper_searcher_kwargs = {
    'n_iter': args.num_hyp_samples,
    'n_jobs': args.num_jobs,
    'verbose': 1,
    'iid': True,
    'return_train_score': False,
    'refit': False,
    'error_score': 'raise'
}
hypotheses = challenge.get_hypotheses(train=train, hyper_searcher_kwargs=hyper_searcher_kwargs,
                                      cv_folds=args.cv_folds, cv_repeats=args.cv_repeats)
print('done\n')


print_title('SEARCHING FOR BEST HYPERPARAMETERS')
assert len(hypotheses) > 0, 'At least one hypothesis must be returned by {c}.get_hypotheses.'.format(c=args.challenge)
for hypothesis_name, hypothesis in hypotheses.items():
    run_time = int(time())

    # TODO - ABSTRACT THIS JAZZ
    # TODO - WHAT ABOUT WHEN HYPER_SEARCHER IS NOT RAND SEARCH AND THIS DOESNT WORK?
    # ==========================
    print('Training hypothesis "{h}_{r}".'.format(r=run_time, h=hypothesis_name))
    y = hypothesis.dataset.target
    y = y.values.ravel() if len(y.columns) == 1 else y.values
    hyper_searcher = hypothesis.hyper_searcher
    if args.num_hyp_samples == 0:
        print('Fitting base model without sampling hyperparameters')
        hyper_searcher.set_params(**{'param_distributions': {}, 'n_iter': 1})
    # ==========================

    hyper_searcher.fit(X=hypothesis.dataset.features, y=y)
    hypothesis.best_model = hyper_searcher.best_estimator_
    hypothesis.best_hyperparams = hyper_searcher.best_params_
    hypothesis.cv_results = process_cv_results(
        run_id=run_time,
        hypothesis_name=hypothesis_name,
        cv_results=hyper_searcher.cv_results_,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
        num_hyp_samples=args.num_hyp_samples
    )
    save_cv_results(
        path='challenges/{c}/results.csv'.format(c=args.challenge),
        processed_cv_results=hypothesis.cv_results,
        top_n=args.report_n,
        reporting_keys=REPORTING_KEYS
    )
    save_hypothesis(
        path='challenges/{c}/hypotheses/{h}_{r}.dump'.format(c=args.challenge, r=run_time, h=hypothesis_name),
        hypothesis=hypothesis
    )
    # TODO: PRINT RESULTS FOR TOP ARGS.REPORT_N MODELS
