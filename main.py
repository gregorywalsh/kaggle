import importlib
import joblib
import csv
from time import time
from argparse import ArgumentParser
from dataset import Dataset
from hypersearch import search_hyperparameter_space


# HELPER FUNCTIONS
def print_title(title):
    many_stars = '*' * 100
    print(many_stars + '\n' + title + '\n' + many_stars)


# PARSE ARGUMENTS
parser = ArgumentParser(description='Train and evaluate bunch of models')
parser.add_argument('-c', '--challenge', type=str, default='titanic',
                    help='Name of challenge folder')
parser.add_argument('-o', '--output_folder', type=str, default='machines/',
                    help='Output folder (with trailing slash)')
parser.add_argument('-F', '--num_folds', type=int, default=4,
                    help='The number of cross validation folds')
parser.add_argument('-H', '--num_hyp_samp_per_hypoth', type=int, default=5,
                    help='The number of hyperparam samples per hypothesis')
parser.add_argument('-I', '--num_hyp_samp_for_best', type=int, default=25,
                    help='The number of hyperparam samples for top performing hypothesis')
parser.add_argument('-J', '--num_jobs', type=int, default=-1,
                    help='The number of parallel jobs to run (-1 all cores)')
parser.add_argument('-R', '--num_rows', type=int, default=None,
                    help='The number of rows or fraction in interval [0,1] to use in training')
args = parser.parse_args()


print_title('LOADING TEST AND TRAIN DATA')
# TODO: Add some validation for test vs training col_defns (or at the fit/predict stage on conduit)
train = Dataset(
    config_fp='challenges/{}/config/train.yml'.format(args.challenge),
    data='challenges/{}/data/train.csv'.format(args.challenge),
    num_rows=args.num_rows,
    verbose=True,
    always_validate=True
)
test = Dataset(
    config_fp='challenges/{}/config/test.yml'.format(args.challenge),
    data='challenges/{}/data/test.csv'.format(args.challenge),
    verbose=True,
    num_rows=None,
    always_validate=True
)
print('{r} training rows and {e} test rows loaded\n'.format(r=train.features.shape[0], e=test.features.shape[0]))


print_title('PERFORMING INITIAL DATA MUNGE')
challenge = importlib.import_module('challenges.{c}.{c}'.format(c=args.challenge))
for dataset in [train, test]:
    dataset.features = challenge.initial_data_munge(dataset.features)
print('done\n')

print_title('BUILDING HYPOTHESES')
hypotheses = challenge.get_hypotheses(train_features=train.features, test_features=test.features)
print('done\n')


print_title('SEARCHING FOR BEST HYPERPARAMETERS')
run_id = int(time())
if len(hypotheses) == 0:
    raise ValueError('You must specify at least one hypothesis in your challenge file.')
elif len(hypotheses) == 1:
    print('Training {m} machines for 1 hypothesis for {k} outer folds. Total: {t}.'.format(
        m=args.num_hyp_samp_per_hypoth, c=len(hypotheses),
        k=args.num_folds, t=args.num_hyp_samp_per_hypoth * args.num_folds)
    )
else:
    print('Training {m} machines for each of {c} conduits for {k} inner and {k} outer folds. Total: {t}.'.format(
        m=args.num_hyp_samp_by_hypoth, c=len(hypotheses),
        k=args.num_folds, t=args.num_hyp_samp_per_hypoth * len(hypotheses) * args.num_folds ** 2)
    )
cv_kwargs = {
    'n_iter': args.num_hyp_samp_per_hypoth,
    'n_jobs': args.num_jobs,
    'verbose': 1,
    'cv': challenge.get_cv()
}
all_cv_results = []
for hypothesis_name, hypothesis in hypotheses.items():
    hypotheses[hypothesis_name].best_hyperparams, cv_results = search_hyperparameter_space(
        features=train.features,
        target=train.target.values.ravel() if len(train.target.columns) == 1 else train.target.values,
        hypothesis=hypothesis,
        scoring=challenge.get_scoring(),
        num_folds=args.num_folds,
        cv_kwargs=cv_kwargs
    )
    all_cv_results.append(cv_results)
with open(file='challenges/{c}/results.csv'.format(c=args.challenge), mode='a+') as f:
    f_writer = csv.writer(f, dialect='excel')
    data = zip(*[cv_results[key] for key in sorted(cv_results.keys())])
    for cv_results in all_cv_results:
        f_writer.writerows(sorted(data))


print_title("EVALUATING TRAINING VARIANCE FOR HYPOTHESES WITH NON-DETERMINISTIC TRAINING")
for hypothesis_name, hypothesis in hypotheses:
    if hypothesis.is_trained_stochastically():
        # TODO implement: evaluate_training_variance(hypothesis)
        pass
    
print_title('GETTING BEST STACKED MODELS')
if len(hypotheses) == 1:
    print('Training {m} machines for 1 hypothesis for {k} folds. Total: {t}.'.format(
        m=args.num_hyp_samp_by_hypoth, k=args.num_folds, t=args.num_hyp_samp_by_hypoth * args.num_folds)
    )
else:
    print('Training {m} machines for each of {c} hypotheses for {k} inner and {k} outer folds. Total: {t}.'.format(
        m=args.num_hyp_samp_by_hypoth, c=len(hypotheses), k=args.num_folds,
        t=args.num_hyp_samp_by_hypoth * len(hypotheses) * args.num_folds ** 2
    ))
best_machine = None
best_rand_search_kwargs = {
    'n_iter': args.num_hyp_samp_for_best,
    'n_jobs': args.num_jobs,
    'verbose': 1
}
print_title('SAVING KERNEL')
filepath = 'challenges/{c}/kernels/{r}.joblib'.format(c=args.challenge, r=run_id)
joblib.dump(best_machine, filename=filepath)
print('newly trained machine saved to "{}"'.format(filepath))

print_title()