import challenges.spooky_author_identification.hypotheses as hyps
import pandas as pd

from datareader import DataReader
from experiment import Experiment
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.base import clone
from scipy.special import softmax
from temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader


# PARAMS
DIR = 'challenges/spooky_author_identification'


# VARIABLES
cv_splits = 2
cv_repeats = 1

# LOAD DATA AND GET EMBEDDING INDEXES
#   Load data
data_reader = DataReader(config_fp='{}/data/config.yml'.format(DIR))
df_train = data_reader.load_from_csv(
    fp='{}/data/train.csv'.format(DIR),
    validate_col_names=True,
    is_test=False,
    append_vartype=False
)
df_test = data_reader.load_from_csv(
    fp='{}/data/test.csv'.format(DIR),
    validate_col_names=True,
    is_test=True,
    append_vartype=False
)

hyper_search_strat = RandomizedSearchCV
hyper_search_kwargs = {
    'cv': RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats),
    'scoring': 'neg_log_loss',
    'error_score': 'raise',
    'refit': True
}
hypotheses = [
    hyps.RNN(
        hyper_search_strat=hyper_search_strat,
        hyper_search_kwargs=hyper_search_kwargs,
    )
]

experiment = Experiment(
    x_train=df_train,
    x_test=df_test,
    y_train=y_train,
    hypotheses=hypotheses
)


# FIT MODEL
clf.fit(X=xs['train'], y=ys['train'])
clf.initialize()
clf.load_params('{c}/models/rnn/params.pt'.format(c=DIR))


# CALIBRATE PROBABILITIES
clf_clone = clone(clf)
cal_model = ModelWithTemperature(model=clf_clone.module_, device=device)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=True)
clf_clone.module_ = cal_model.set_temperature(valid_loader=val_loader)


# WRITE OUT PREDICTIONS TO FILE
df_pred = pd.DataFrame(data=clf_clone.predict_proba(X=xs['test']), columns=TARGET_COLUMNS)
df_pred[TARGET_COLUMNS] = softmax(x=df_pred[TARGET_COLUMNS], axis=1)
df_pred['id'] = df_all['id']['test']
df_pred[['id'] + TARGET_COLUMNS].to_csv(path_or_buf='{c}/predictions/submission_temp.csv'.format(c=DIR), index=False)


# CROSS VALIDATE
def negative_log_loss(y_true, y_pred_raw):
    y_pred = softmax(x=y_pred_raw, axis=1)
    return -log_loss(y_true=y_true, y_pred=y_pred)


nll_scorer = make_scorer(score_func=negative_log_loss, greater_is_better=False, needs_proba=True)
