import numpy as np
import importlib
from time import time
from argparse import ArgumentParser
from misc.old.dataset import Dataset
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from util import print_title, load_yaml_config


# PARSE ARGUMENTS
parser = ArgumentParser(description='Train and evaluate bunch of checkpoint')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
args = parser.parse_args()


# MAIN SCRIPT
print_title('LOADING CONFIG')
config_fp = 'challenges/{}/data/config.yml'.format(args.challenge)
config = load_yaml_config(fp=config_fp)
print('Done')


print_title('LOADING TEST AND TRAINING DATA')
print('Challenge {c}\n'.format(c=args.challenge))
test = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/test.csv'.format(args.challenge),
    is_test=True,
    verbose=True,
    num_rows=None,
    always_validate=True
)
train = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/train.csv'.format(args.challenge),
    is_test=False,
    verbose=True,
    num_rows=None,
    always_validate=True
)

print_title('MUNGING DATA')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
test.features = challenge.munge_features(test.features)
train.features = challenge.munge_features(train.features)
print('done\n')


print_title('BUILDING HYPOTHESES')
hypotheses = challenge.get_hypotheses(train=train, hyper_searcher_kwargs={},
                                      cv_folds=1, cv_repeats=1)
hypothesis = hypotheses['rf']
print('done\n')

print_title('CROSS VALIDATING')
# base_clfs = []
# for i in range(0, 25):
#     base_clfs += [(str(i), hypothesis.model)]
# clf = VotingClassifier(estimators=base_clfs, n_jobs=args.num_jobs)

results = cross_val_score(
    estimator=hypothesis.model,
    X=train.features,
    y=train.target.values.ravel() if len(train.target.columns) == 1 else train.target.values,
    scoring='accuracy',
    cv=RepeatedStratifiedKFold(25, 2),
    verbose=1,
    n_jobs=args.num_jobs
)
print(results)
print(sum(results)/len(results))
print('done\n')

print_title('MAKING PREDICTIONS')
y_train = train.target.values.ravel() if len(train.target.columns) == 1 else train.target.values
hypothesis.model.fit(X=train.features, y=y_train)

run_time = int(time())
predictions = hypothesis.model.predict(X=test.features)
submission = np.column_stack((test.meta['passenger_id|unique'].astype('int').to_numpy(), predictions))
np.savetxt(
    fname='challenges/{c}/predictions/{r}.csv'.format(c=args.challenge, r=run_time),
    X=submission,
    fmt=config['submission_format'],
    header=config['submission_header'],
    comments='',
    delimiter=",")
print('done\n')
