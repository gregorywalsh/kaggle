import importlib
from argparse import ArgumentParser
from misc.old.dataset import Dataset
from hypothesis import load_hypothesis
from util import print_title, load_yaml_config
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


# PARSE ARGS
# ======================================================================================================================
parser = ArgumentParser(description='Train and evaluate bunch of checkpoint')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-r', '--run_id', type=str, default='rf_1563389884',
                    help='The run id of the hypothesis to load')
parser.add_argument('-K', '--cv_folds', type=int, default=10,
                    help='The number of CV folds')
parser.add_argument('-P', '--cv_repeats', type=int, default=2,
                    help='The number of CV repeats')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
args = parser.parse_args()


# MAIN
# ======================================================================================================================
print_title('LOADING CONFIG')
config_fp = 'challenges/{}/data/config.yml'.format(args.challenge)
config = load_yaml_config(fp=config_fp)
print('Done')


print_title('LOADING HYPOTHESIS AND TRAINING DATA')
print('Challenge {c}, run id {r}\n'.format(c=args.challenge, r=args.run_id))
hypothesis = load_hypothesis(path='challenges/{c}/saved_hypotheses/{r}.dump'.format(c=args.challenge, r=args.run_id),)
train = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/train.csv'.format(args.challenge),
    is_test=False,
    verbose=True,
    num_rows=None,
    always_validate=True
)


print_title('MUNGING TRAINING DATA')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
train.features = challenge.munge_features(train.features)
print('done\n')


print_title('CROSS VALIDATING')
results = cross_val_score(
    estimator=hypothesis.best_model,
    X=train.features,
    y=train.target.values.ravel() if len(train.target.columns) == 1 else train.target.values,
    scoring='accuracy',
    cv=RepeatedStratifiedKFold(args.cv_folds, args.cv_repeats),
    verbose=1,
    n_jobs=args.num_jobs
)
print(results)
print(sum(results)/len(results))
