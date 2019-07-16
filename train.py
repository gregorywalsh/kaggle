import importlib
from time import time
from argparse import ArgumentParser
from dataset import Dataset
from hypersearch import search_hyperparam_space, process_cv_results, save_cv_results
from hypotheses import save_hypothesis
from util import print_title


# STATIC VARS
CV_REPORTING_KEYS = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'params']


# PARSE ARGUMENTS
parser = ArgumentParser(description='Train and evaluate bunch of models')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-o', '--output_folder', type=str, default='machines/',
                    help='Output folder (with trailing slash)')
parser.add_argument('-H', '--num_hyp_samp_per_hypoth', type=int, default=5,
                    help='The number of hyperparam samples per hypothesis')
parser.add_argument('-I', '--num_hyp_samp_for_best', type=int, default=25,
                    help='The number of hyperparam samples for top performing hypothesis')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
parser.add_argument('-R', '--num_rows', type=int, default=None,
                    help='The number of rows or fraction in interval [0,1] to use in training')
parser.add_argument('-T', '--report_n', type=int, default=5,
                    help='The number of models on which to report training performance')
args = parser.parse_args()


# MAIN SCRIPT
print_title('BEGINNING RUN')
print('Challenge {c}\n'.format(c=args.challenge))


print_title('LOADING TEST AND TRAIN DATA')
# TODO: Add some validation for test vs training col_defns (or at the fit/predict stage on conduit)
train = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data='challenges/{}/data/train.csv'.format(args.challenge),
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
    'n_iter': args.num_hyp_samp_per_hypoth,
    'n_jobs': args.num_jobs,
    'verbose': 1,
    'iid': True,
    'return_train_score': False,
    'refit': False,
    'error_score': 'raise'
}
hypotheses = challenge.get_hypotheses(train=train, hyper_searcher_kwargs=hyper_searcher_kwargs)
print('done\n')


print_title('SEARCHING FOR BEST HYPERPARAMETERS')
assert len(hypotheses) > 0, 'You must specify at least one hypothesis in your challenge file.'
all_cv_results = []
for hypothesis_name, hypothesis in hypotheses.items():
    run_time = int(time())
    print('Training hypothesis "{h}_{r}".'.format(r=run_time, h=hypothesis_name))
    hypothesis.best_model, hypothesis.best_hyperparams, raw_cv_results = search_hyperparam_space(
        hypothesis=hypothesis,
    )
    hypothesis.cv_results = process_cv_results(
        run_id=run_time,
        hypothesis_name=hypothesis_name,
        cv_results=raw_cv_results,
        cv_reporting_keys=CV_REPORTING_KEYS,
    )
    save_cv_results(
        path='challenges/{c}/results.csv'.format(c=args.challenge),
        processed_cv_results=hypothesis.cv_results,
        top_n=args.report_n
    )
    save_hypothesis(
        path='challenges/{c}/hypotheses/{h}_{r}.dump'.format(c=args.challenge, r=run_time, h=hypothesis_name),
        hypothesis=hypothesis
    )
    # TODO: PRINT RESULTS FOR TOP ARGS.REPORT_N MODELS
