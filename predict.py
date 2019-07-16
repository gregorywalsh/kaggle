import numpy as np
import importlib
from argparse import ArgumentParser
from dataset import Dataset
from hypotheses import load_hypothesis
from util import print_title


# PARSE ARGUMENTS
parser = ArgumentParser(description='Train and evaluate bunch of models')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-r', '--run_id', type=str, default='rf_1563311268.dump',
                    help='The number of parallel jobs to run (-1 all cores)')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
args = parser.parse_args()


# MAIN SCRIPT
print_title('LOADING HYPOTHESIS AND TRAINING DATA')
print('Challenge {c}, run id {r}\n'.format(c=args.challenge, r=args.run_id))
hypothesis = load_hypothesis(path='challenges/{c}/hypotheses/{r}'.format(c=args.challenge, r=args.run_id),)
test = Dataset(
    config_fp='challenges/{}/data/config.yml'.format(args.challenge),
    data='challenges/{}/data/test.csv'.format(args.challenge),
    is_test=True,
    verbose=True,
    num_rows=None,
    always_validate=True
)


print_title('MUNGING TEST DATA')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
test.features = challenge.munge_features(test.features)
print('done\n')


print_title('MAKING PREDICTIONS')
predictions = hypothesis.best_model.predict(test.features).astype('int')
submission = np.column_stack((test.meta['passenger_id|unique'].astype('int').to_numpy(), predictions))
np.savetxt('challenges/{c}/predictions/{r}.csv'.format(c=args.challenge, r=args.run_id), submission, delimiter=",")
print('done\n')
