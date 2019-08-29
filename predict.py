import numpy as np
import importlib
from argparse import ArgumentParser
from dataset import Dataset
from hypothesis import load_hypothesis
from util import print_title, load_yaml_config


# PARSE ARGS
# ======================================================================================================================
parser = ArgumentParser(description='Train and evaluate bunch of checkpoint')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-r', '--run_id', type=str, default='lgb_1563476819',
                    help='The run id of the hypothesis to load')
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
hypothesis = load_hypothesis(path='challenges/{c}/saved_hypotheses/{r}.dump'.format(c=args.challenge, r=args.run_id))
train = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/train.csv'.format(args.challenge),
    is_test=False,
    num_rows=None,
    verbose=True,
    always_validate=True
)
test = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data_fp='challenges/{}/data/test.csv'.format(args.challenge),
    is_test=True,
    verbose=True,
    num_rows=None,
    always_validate=True
)

print_title('MUNGING TEST DATA')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
_, test.features = challenge.munge_features(train.features, test.features)
print('done\n')


print_title('MAKING PREDICTIONS')
predictions = hypothesis.best_model.predict(test.features)
submission = np.column_stack((test.meta['passenger_id|unique'].astype('int').to_numpy(), predictions))
np.savetxt(
    fname='challenges/{c}/predictions/{r}.csv'.format(c=args.challenge, r=args.run_id),
    X=submission,
    fmt=config['submission_format'],
    header=config['submission_header'],
    comments='',
    delimiter=",")
print('done\n')
